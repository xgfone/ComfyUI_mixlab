from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from xml.etree import ElementTree

import requests
from typing_extensions import Self

ACS3_ALGORITHM = "ACS3-HMAC-SHA256"
FACEBODY_API_VERSION = "2019-12-30"
OPEN_PLATFORM_API_VERSION = "2019-12-19"
OPEN_PLATFORM_ENDPOINT = "openplatform.aliyuncs.com"
DEFAULT_REGION_ID = "cn-shanghai"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024

_SUCCESS_CODES = {"200", "OK", "Success"}
_REGION_PATTERN = re.compile(r"^[a-z0-9-]+$")
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$")
_HOST_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")


class AliyunFaceBodyError(RuntimeError):
    """Base error raised by the dependency-free Aliyun HTTP client."""


class AliyunFaceBodyAPIError(AliyunFaceBodyError):
    """An error response returned by an Alibaba Cloud OpenAPI endpoint."""

    def __init__(
        self,
        *,
        action: str,
        status_code: int,
        code: Any = None,
        message: Any = None,
        request_id: Any = None,
        response_text: str = "",
    ) -> None:
        self.action = action
        self.status_code = status_code
        self.code = code
        self.message = message
        self.request_id = request_id
        self.response_text = response_text

        parts = [f"aliyun {action} request failed (HTTP {status_code})"]
        if code not in (None, ""):
            parts.append(f"Code={code}")
        if message not in (None, ""):
            parts.append(f"Message={message}")
        if request_id not in (None, ""):
            parts.append(f"RequestId={request_id}")
        if len(parts) == 1 and response_text:
            parts.append(f"Response={_truncate(response_text)}")
        super().__init__(", ".join(parts))


@dataclass(frozen=True)
class AliyunCredentials:
    access_key_id: str
    access_key_secret: str
    security_token: str | None = None


def resolve_aliyun_credentials(
    access_key_id: str = "", access_key_secret: str = ""
) -> AliyunCredentials:
    """
    Resolve credentials from node inputs, then standard and legacy env vars.

    The legacy names keep workflows made for the removed SDK node working.
    """
    supplied_id = (access_key_id or "").strip()
    supplied_secret = (access_key_secret or "").strip()

    if supplied_id and supplied_secret:
        resolved_id = supplied_id
        resolved_secret = supplied_secret
    else:
        standard_id = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_ID", "").strip()
        standard_secret = os.environ.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", "").strip()
        legacy_id = os.environ.get("ALIYUN_APPKEY", "").strip()
        legacy_secret = os.environ.get("ALIYUN_SECRET", "").strip()
        if standard_id and standard_secret:
            resolved_id, resolved_secret = standard_id, standard_secret
        elif legacy_id and legacy_secret:
            resolved_id, resolved_secret = legacy_id, legacy_secret
        else:
            # Never combine half of one credential pair with half of another.
            resolved_id, resolved_secret = "", ""

    if not resolved_id or not resolved_secret:
        raise ValueError(
            "Missing Alibaba Cloud AccessKey. Please fill in the access_key_id/access_key_secret of the node, "
            "or set the ALIBABA_CLOUD_ACCESS_KEY_ID and ALIBABA_CLOUD_ACCESS_KEY_SECRET environment variables."
        )

    security_token = os.environ.get("ALIBABA_CLOUD_SECURITY_TOKEN", "").strip() or None
    return AliyunCredentials(resolved_id, resolved_secret, security_token)


class AliyunFaceBodyHTTPClient:
    """
    Minimal requests-based client for FaceBeauty and its temporary OSS upload.

    It implements Alibaba Cloud OpenAPI signature V3 directly and deliberately
    has no dependency on any alibabacloud_* package.
    """

    def __init__(
        self,
        credentials: AliyunCredentials,
        *,
        region_id: str = DEFAULT_REGION_ID,
        read_timeout: float = DEFAULT_READ_TIMEOUT_SECONDS,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        session: requests.Session | None = None,
    ) -> None:
        region_id = (region_id or "").strip()
        if not _REGION_PATTERN.fullmatch(region_id):
            raise ValueError(f"Invalid Alibaba Cloud RegionId: {region_id!r}")
        if not credentials.access_key_id or not credentials.access_key_secret:
            raise ValueError("access_key_id and access_key_secret cannot be empty.")

        self.credentials = credentials
        self.region_id = region_id
        self.facebody_endpoint = f"facebody.{region_id}.aliyuncs.com"
        self.timeout = (float(connect_timeout), float(read_timeout))
        self._session = session or requests.Session()
        self._owns_session = session is None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    def upload_image(
        self,
        image_bytes: bytes,
        *,
        content_type: str = "image/png",
    ) -> str:
        """Upload a local image with the temporary form returned by Aliyun."""
        if not isinstance(image_bytes, (bytes, bytearray)) or not image_bytes:
            raise ValueError("The image content to be uploaded cannot be empty.")

        authorization = self._call_rpc_json(
            endpoint=OPEN_PLATFORM_ENDPOINT,
            action="AuthorizeFileUpload",
            version=OPEN_PLATFORM_API_VERSION,
            method="GET",
            query={"Product": "facebody", "RegionId": self.region_id},
        )

        bucket = _require_string(authorization, "Bucket", "AuthorizeFileUpload")
        endpoint = _require_string(authorization, "Endpoint", "AuthorizeFileUpload")
        object_key = _require_string(authorization, "ObjectKey", "AuthorizeFileUpload")
        access_key_id = _require_string(authorization, "AccessKeyId", "AuthorizeFileUpload")
        encoded_policy = _require_string(authorization, "EncodedPolicy", "AuthorizeFileUpload")
        signature = _require_string(authorization, "Signature", "AuthorizeFileUpload")

        if not _BUCKET_PATTERN.fullmatch(bucket):
            raise AliyunFaceBodyError(
                f"AuthorizeFileUpload returned an invalid OSS Bucket: {bucket!r}"
            )

        use_accelerate = _as_bool(authorization.get("UseAccelerate"))
        upload_endpoint = (
            "oss-accelerate.aliyuncs.com" if use_accelerate else _normalize_host(endpoint)
        )
        upload_url = f"https://{bucket}.{upload_endpoint}/"
        form = {
            "OSSAccessKeyId": access_key_id,
            "policy": encoded_policy,
            "Signature": signature,
            "key": object_key,
            "success_action_status": "201",
        }

        try:
            response = self._session.post(
                upload_url,
                data=form,
                # The generated Aliyun SDK uses ObjectKey as the multipart
                # filename as well as the separate key form field.
                files={"file": (object_key, bytes(image_bytes), content_type)},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AliyunFaceBodyError(
                f"Failed to upload image to temporary Alibaba Cloud OSS: {exc}"
            ) from exc

        try:
            if not 200 <= response.status_code < 300:
                code, message, request_id = _parse_oss_error(response.text)
                raise AliyunFaceBodyAPIError(
                    action="OSSUpload",
                    status_code=response.status_code,
                    code=code,
                    message=message,
                    request_id=request_id,
                    response_text=response.text,
                )
        finally:
            response.close()

        # Match the official SDK: the URL used by FaceBeauty always points at
        # the regional endpoint, even when the upload itself used acceleration.
        encoded_key = quote(object_key.lstrip("/"), safe="/~")
        return f"https://{bucket}.{endpoint}/{encoded_key}"

    def face_beauty(self, image_url: str, *, sharp: float, smooth: float, white: float) -> str:
        image_url = _validate_http_url(image_url, "ImageURL")
        sharp = _validate_strength(sharp, "Sharp")
        smooth = _validate_strength(smooth, "Smooth")
        white = _validate_strength(white, "White")

        payload = self._call_rpc_json(
            endpoint=self.facebody_endpoint,
            action="FaceBeauty",
            version=FACEBODY_API_VERSION,
            method="POST",
            form={
                "ImageURL": image_url,
                "Sharp": sharp,
                "Smooth": smooth,
                "White": white,
            },
        )
        data = payload.get("Data")
        if not isinstance(data, Mapping):
            raise AliyunFaceBodyError("FaceBeauty response is missing the Data object.")
        return _validate_http_url(
            _require_string(data, "ImageURL", "FaceBeauty.Data"), "Result ImageURL"
        )

    def download_image(
        self, image_url: str, *, max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES
    ) -> bytes:
        image_url = _validate_http_url(image_url, "Result ImageURL")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be a positive number.")

        try:
            response = self._session.get(image_url, stream=True, timeout=self.timeout)
        except requests.RequestException as exc:
            raise AliyunFaceBodyError(
                f"Failed to download Alibaba Cloud FaceBeauty result: {exc}"
            ) from exc

        try:
            if not 200 <= response.status_code < 300:
                raise AliyunFaceBodyAPIError(
                    action="DownloadResult",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise AliyunFaceBodyError(
                            f"Alibaba Cloud FaceBeauty result exceeds download size limit: {content_length} > {max_bytes} bytes."
                        )
                except ValueError:
                    pass

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise AliyunFaceBodyError(
                        f"Alibaba Cloud FaceBeauty result exceeds download size limit: {max_bytes} bytes."
                    )
                chunks.append(chunk)
            if not chunks:
                raise AliyunFaceBodyError(
                    "Alibaba Cloud FaceBeauty result downloaded successfully, but the response content is empty."
                )
            return b"".join(chunks)
        finally:
            response.close()

    def _call_rpc_json(
        self,
        *,
        endpoint: str,
        action: str,
        version: str,
        method: str,
        query: Mapping[str, Any] | None = None,
        form: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        method = method.upper()
        query = query or {}
        body = _form_encode(form) if form is not None else b""
        content_type = "application/x-www-form-urlencoded" if form is not None else None
        headers = _build_acs3_headers(
            method=method,
            host=endpoint,
            action=action,
            version=version,
            query=query,
            body=body,
            access_key_id=self.credentials.access_key_id,
            access_key_secret=self.credentials.access_key_secret,
            security_token=self.credentials.security_token,
            content_type=content_type,
        )
        headers["Accept"] = "application/json"

        canonical_query = _canonical_query_string(query)
        url = f"https://{endpoint}/"
        if canonical_query:
            url = f"{url}?{canonical_query}"

        try:
            response = self._session.request(
                method,
                url,
                headers=headers,
                data=body if form is not None else None,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise AliyunFaceBodyError(f"Alibaba Cloud {action} request failed: {exc}") from exc

        try:
            response_text = response.text
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                if not 200 <= response.status_code < 300:
                    raise AliyunFaceBodyAPIError(
                        action=action,
                        status_code=response.status_code,
                        response_text=response_text,
                    ) from exc
                raise AliyunFaceBodyError(
                    f"Alibaba Cloud {action} returned invalid JSON: {_truncate(response_text)}"
                ) from exc

            if not isinstance(payload, dict):
                raise AliyunFaceBodyError(
                    f"Alibaba Cloud {action} returned JSON with a non-object top level."
                )

            code = payload.get("Code", payload.get("code"))
            is_error_code = code not in (None, "") and str(code) not in _SUCCESS_CODES
            if not 200 <= response.status_code < 300 or is_error_code:
                raise AliyunFaceBodyAPIError(
                    action=action,
                    status_code=response.status_code,
                    code=code,
                    message=payload.get("Message", payload.get("message")),
                    request_id=payload.get("RequestId", payload.get("requestId")),
                    response_text=response_text,
                )
            return payload
        finally:
            response.close()


def _build_acs3_headers(
    *,
    method: str,
    host: str,
    action: str,
    version: str,
    query: Mapping[str, Any],
    body: bytes,
    access_key_id: str,
    access_key_secret: str,
    security_token: str | None = None,
    content_type: str | None = None,
    acs_date: str | None = None,
    signature_nonce: str | None = None,
) -> dict[str, str]:
    """Build ACS3-HMAC-SHA256 headers. Optional time/nonce make it testable."""
    payload_hash = hashlib.sha256(body).hexdigest()
    headers: dict[str, str] = {
        "host": host,
        "x-acs-action": action,
        "x-acs-content-sha256": payload_hash,
        "x-acs-date": acs_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "x-acs-signature-nonce": signature_nonce or uuid.uuid4().hex,
        "x-acs-version": version,
    }
    if content_type:
        headers["content-type"] = content_type
    if security_token:
        headers["x-acs-security-token"] = security_token

    signed_items = sorted((key.lower(), str(value).strip()) for key, value in headers.items())
    canonical_headers = "".join(f"{key}:{value}\n" for key, value in signed_items)
    signed_headers = ";".join(key for key, _ in signed_items)
    canonical_request = "\n".join(
        [
            method.upper(),
            "/",
            _canonical_query_string(query),
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )
    canonical_request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = f"{ACS3_ALGORITHM}\n{canonical_request_hash}"
    signature = hmac.new(
        access_key_secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    headers["Authorization"] = (
        f"{ACS3_ALGORITHM} Credential={access_key_id},"
        f"SignedHeaders={signed_headers},Signature={signature}"
    )
    return headers


def _canonical_query_string(query: Mapping[str, Any]) -> str:
    items = []
    for key, value in query.items():
        if value is None:
            continue
        encoded_key = quote(str(key), safe="~")
        encoded_value = quote(str(value), safe="~")
        items.append((encoded_key, encoded_value))
    items.sort(key=lambda item: item[0])
    return "&".join(f"{key}={value}" for key, value in items)


def _form_encode(form: Mapping[str, Any]) -> bytes:
    values = [(str(key), str(value)) for key, value in form.items() if value is not None]
    return urlencode(values).encode("utf-8")


def _validate_http_url(value: str, name: str) -> str:
    value = (value or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a valid http/https URL.")
    if any(ord(char) > 127 for char in value):
        raise ValueError(f"{name} cannot contain unencoded non-ASCII characters.")
    return value


def _validate_strength(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between 0 and 1.") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def _require_string(payload: Mapping[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AliyunFaceBodyError(f"{context} response is missing a non-empty string field {key}.")
    return value.strip()


def _normalize_host(value: str) -> str:
    candidate = value.strip()
    if "://" in candidate:
        candidate = urlparse(candidate).netloc
    candidate = candidate.strip().strip("/").rstrip(".")
    if not candidate or ":" in candidate or not _HOST_PATTERN.fullmatch(candidate):
        raise AliyunFaceBodyError(
            f"AuthorizeFileUpload returned an invalid OSS Endpoint: {value!r}"
        )
    return candidate


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_oss_error(
    response_text: str,
) -> tuple[str | None, str | None, str | None]:
    try:
        root = ElementTree.fromstring(response_text)
    except ElementTree.ParseError:
        return None, _truncate(response_text) if response_text else None, None

    def text(name: str) -> str | None:
        element = root.find(f".//{name}")
        return element.text if element is not None else None

    return text("Code"), text("Message"), text("RequestId")


def _truncate(value: str, limit: int = 2000) -> str:
    value = (value or "").strip()
    return value if len(value) <= limit else f"{value[:limit]}…"
