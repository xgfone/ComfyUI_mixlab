from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests

DEFAULT_MODEL_ID = "default_segment"
DEFAULT_BASE_URL = "https://api.bimoai.com"
DEFAULT_POLL_INTERVAL_SECONDS = 2.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 60.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 3.0

SUBMIT_PATH = "/segment/image/v1/task/submit"
QUERY_PATH = "/segment/image/v1/task/query"

PENDING_STATUSES = {"Queuing", "Processing"}
TERMINAL_ERROR_STATUSES = {"Failure", "NotFound"}
TRANSIENT_STATUSES = {"ServerError"}
KNOWN_STATUSES = PENDING_STATUSES | TERMINAL_ERROR_STATUSES | TRANSIENT_STATUSES | {"Success"}


class BimoAISegmentError(RuntimeError):
    """Base exception surfaced by ComfyUI when this node fails."""


class BimoAIHTTPError(BimoAISegmentError):
    """HTTP error where the server did not provide a wrapped Error object."""

    def __init__(self, status_code: int, method: str, url: str, response_text: str) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.response_text = response_text

        msg = f"BimoAI HTTP 请求失败：{method} {url} -> HTTP {status_code}；响应：{_truncate(response_text, 2000)}"
        super().__init__(msg)


class BimoAIAPIError(BimoAISegmentError):
    """Error returned in the wrapped response's top-level Error field."""

    def __init__(
        self,
        *,
        status_code: int,
        method: str,
        url: str,
        code: Any,
        reason: Any,
        message: Any,
        response_payload: Dict[str, Any],
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.code = code
        self.reason = reason
        self.message = message
        self.response_payload = response_payload

        parts = [f"BimoAI API 请求失败：{method} {url} -> HTTP {status_code}"]
        if code not in (None, ""):
            parts.append(f"Code={code}")
        if reason not in (None, ""):
            parts.append(f"Reason={reason}")
        if message not in (None, ""):
            parts.append(f"Message={message}")
        if len(parts) == 1:
            parts.append(f"Error={_safe_json(response_payload.get('Error'))}")
        super().__init__("；".join(parts))


@dataclass(frozen=True)
class ValidatedInputs:
    base_url: str
    model_id: str
    image_url: str
    max_persons: int
    poll_interval_seconds: float
    http_timeout_seconds: float
    timeout_seconds: float


class BimoAISegmentImage:
    """
    Submit a BimoAI segmentation task, then poll its status until success,
    terminal failure, or the workflow-configured total timeout is reached.

    Only the authentication token is read from the environment:
      BIMOAI_TOKEN

    Every other configurable value comes from the node's workflow inputs.
    """

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("result_image_url", "task_id", "response_json")
    FUNCTION = "segment_image"
    CATEGORY = "BimoAI/API"
    DESCRIPTION = "提交 BimoAI 抠图任务并轮询 wrapped 格式结果。"
    SEARCH_ALIASES = ["bimoai segment", "抠图", "background removal"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": DEFAULT_BASE_URL,
                        "placeholder": DEFAULT_BASE_URL,
                        "tooltip": "BimoAI API BaseUrl。默认 https://api.bimoai.com，可在工作流中覆盖。",
                    },
                ),
                "model_id": (
                    "STRING",
                    {
                        "default": DEFAULT_MODEL_ID,
                        "placeholder": DEFAULT_MODEL_ID,
                        "tooltip": "提交接口请求体中的 ModelId。",
                    },
                ),
                "image_url": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "被抠图的图片URL",
                        "tooltip": "提交接口请求体中的 ImageUrl。",
                    },
                ),
                "max_persons": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 6,
                        "step": 1,
                        "tooltip": "提交接口请求体中的 MaxPersons；0 表示由接口按默认规则处理。",
                    },
                ),
                "poll_interval_seconds": (
                    "FLOAT",
                    {
                        "default": DEFAULT_POLL_INTERVAL_SECONDS,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "round": 0.1,
                        "tooltip": "两次状态查询之间的等待间隔，单位为秒。",
                    },
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {
                        "default": DEFAULT_TOTAL_TIMEOUT_SECONDS,
                        "min": 0.1,
                        "max": 600.0,
                        "step": 0.5,
                        "round": 0.1,
                        "tooltip": "从成功取得 TaskId 开始计算的最大轮询总时长，单位为秒。",
                    },
                ),
                "http_timeout_seconds": (
                    "FLOAT",
                    {
                        "default": DEFAULT_HTTP_TIMEOUT_SECONDS,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.5,
                        "round": 0.1,
                        "tooltip": "单次 HTTP 请求的最长等待时长。查询阶段仍受 timeout_seconds 总时长限制。",
                    },
                ),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The result comes from an external API. Re-running the workflow should
        # submit a new task instead of reusing ComfyUI's cached output.
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        image_url: Optional[str] = None,
        max_persons: Optional[int] = None,
        poll_interval_seconds: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
        http_timeout_seconds: Optional[float] = None,
    ):
        try:
            # ComfyUI uses None here for inputs connected to another node,
            # because their runtime values do not exist during prompt
            # validation. Validate only the constants that are available now;
            # segment_image() validates the complete resolved input set again.
            _validate_available_inputs(
                base_url=base_url,
                model_id=model_id,
                image_url=image_url,
                max_persons=max_persons,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
                http_timeout_seconds=http_timeout_seconds,
            )
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    def segment_image(
        self,
        base_url: str,
        model_id: str,
        image_url: str,
        max_persons: int,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
        http_timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> Tuple[str, str, str]:
        inputs = _validate_inputs(
            base_url=base_url,
            model_id=model_id,
            image_url=image_url,
            max_persons=max_persons,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
            http_timeout_seconds=http_timeout_seconds,
        )

        token = _get_token_from_environment()
        headers = _build_headers(token)
        submit_url = _join_url(inputs.base_url, SUBMIT_PATH)
        query_url = _join_url(inputs.base_url, QUERY_PATH)

        with self._new_session() as session:
            submit_payload = {
                "ModelId": inputs.model_id,
                "ImageUrl": inputs.image_url,
                "MaxPersons": inputs.max_persons,
            }

            try:
                submit_response = self._request_wrapped_json(
                    session=session,
                    method="POST",
                    url=submit_url,
                    headers=headers,
                    timeout_seconds=inputs.http_timeout_seconds,
                    json_body=submit_payload,
                )
            except requests.RequestException as exc:
                # Never auto-retry submission, to avoid creating duplicate tasks.
                msg = f"提交抠图任务时发生网络错误：{type(exc).__name__}: {exc}"
                raise BimoAISegmentError(msg) from exc

            data = _require_wrapped_data(submit_response, context="抠图任务提交接口")
            task_id = _require_nonempty_string(data, "TaskId", context="抠图任务提交接口的 Data")

            # The requested total timeout starts only after TaskId has been
            # obtained successfully from the submit API.
            deadline = self._monotonic() + inputs.timeout_seconds
            last_status: Optional[str] = None
            last_response: Any = submit_response
            last_transient_error: Optional[str] = None
            query_count = 0

            while True:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    self._raise_poll_timeout(
                        task_id=task_id,
                        timeout_seconds=inputs.timeout_seconds,
                        query_count=query_count,
                        last_status=last_status,
                        last_response=last_response,
                        last_transient_error=last_transient_error,
                    )

                query_count += 1
                request_timeout = max(0.05, min(inputs.http_timeout_seconds, remaining))

                try:
                    query_response = self._request_wrapped_json(
                        session=session,
                        method="GET",
                        url=query_url,
                        headers=headers,
                        timeout_seconds=request_timeout,
                        query_params={"TaskId": task_id},
                    )
                    last_response = query_response
                    last_transient_error = None
                except BimoAIAPIError as exc:
                    # A wrapped server-side 5xx error during polling is
                    # transient. Other wrapped API errors are terminal.
                    if exc.status_code >= 500:
                        last_transient_error = str(exc)
                        query_response = None
                    else:
                        raise
                except BimoAIHTTPError as exc:
                    # Query-side 5xx errors are transient. Authentication and
                    # other 4xx errors should fail immediately.
                    if exc.status_code >= 500:
                        last_transient_error = str(exc)
                        query_response = None
                    else:
                        raise
                except requests.RequestException as exc:
                    # Network errors while querying are transient until the
                    # workflow's total timeout is reached.
                    last_transient_error = (
                        f"查询任务状态时发生网络错误：{type(exc).__name__}: {exc}"
                    )
                    query_response = None

                # Enforce the total timeout even if a slow request completes
                # after the deadline.
                if self._monotonic() >= deadline:
                    self._raise_poll_timeout(
                        task_id=task_id,
                        timeout_seconds=inputs.timeout_seconds,
                        query_count=query_count,
                        last_status=last_status,
                        last_response=last_response,
                        last_transient_error=last_transient_error,
                    )

                if query_response is not None:
                    query_data = _require_wrapped_data(query_response, context="抠图任务查询接口")
                    status = _require_nonempty_string(
                        query_data, "Status", context="抠图任务查询接口的 Data"
                    )
                    if status not in KNOWN_STATUSES:
                        raise BimoAISegmentError(
                            f"任务查询接口返回未知 Status：{status}；"
                            f"TaskId={task_id}；完整响应：{_safe_json(query_response)}"
                        )

                    last_status = status
                    if status == "Success":
                        result_data = query_data.get("Data")
                        if not isinstance(result_data, dict):
                            raise BimoAISegmentError(
                                "任务状态为 Success，但 Data.Data 不是对象；"
                                f"TaskId={task_id}；完整响应：{_safe_json(query_response)}"
                            )

                        result_url = _require_http_url(
                            result_data,
                            "ImageUrl",
                            context="抠图任务查询接口的 Data.Data",
                        )

                        return (
                            result_url,
                            task_id,
                            json.dumps(query_response, ensure_ascii=False, indent=2),
                        )

                    if status in TERMINAL_ERROR_STATUSES:
                        raise BimoAISegmentError(
                            f"抠图任务失败：Status={status}；TaskId={task_id}；"
                            f"完整响应：{_safe_json(query_response)}"
                        )

                    # Queuing / Processing / ServerError continue polling.

                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    self._raise_poll_timeout(
                        task_id=task_id,
                        timeout_seconds=inputs.timeout_seconds,
                        query_count=query_count,
                        last_status=last_status,
                        last_response=last_response,
                        last_transient_error=last_transient_error,
                    )
                self._sleep(min(inputs.poll_interval_seconds, remaining))

    def _new_session(self) -> requests.Session:
        return requests.Session()

    def _monotonic(self) -> float:
        return time.monotonic()

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _request_wrapped_json(
        self,
        *,
        session: requests.Session,
        method: str,
        url: str,
        headers: Dict[str, str],
        timeout_seconds: float,
        json_body: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        response = session.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            params=query_params,
            timeout=timeout_seconds,
        )

        try:
            payload = response.json()
        except ValueError as exc:
            if not 200 <= response.status_code < 300:
                raise BimoAIHTTPError(
                    status_code=response.status_code,
                    method=method,
                    url=url,
                    response_text=response.text,
                ) from exc

            msg = f"BimoAI 接口返回了非 JSON 内容：{method} {url}；响应：{_truncate(response.text, 2000)}"
            raise BimoAISegmentError(msg) from exc

        if not isinstance(payload, dict):
            msg = f"BimoAI wrapped 应答体必须是 JSON 对象：{method} {url}；完整响应：{_safe_json(payload)}"
            raise BimoAISegmentError(msg)

        error = payload.get("Error")
        if error not in (None, {}, []):
            if isinstance(error, dict):
                code = error.get("Code")
                reason = error.get("Reason")
                message = error.get("Message")
            else:
                code = None
                reason = None
                message = error
            raise BimoAIAPIError(
                status_code=response.status_code,
                method=method,
                url=url,
                code=code,
                reason=reason,
                message=message,
                response_payload=payload,
            )

        if not 200 <= response.status_code < 300:
            raise BimoAIHTTPError(
                status_code=response.status_code,
                method=method,
                url=url,
                response_text=response.text,
            )

        if "Data" not in payload:
            msg = f"BimoAI wrapped 成功应答缺少顶层 Data：{method} {url}；完整响应：{_safe_json(payload)}"
            raise BimoAISegmentError(msg)

        return payload

    @staticmethod
    def _raise_poll_timeout(
        *,
        task_id: str,
        timeout_seconds: float,
        query_count: int,
        last_status: Optional[str],
        last_response: Any,
        last_transient_error: Optional[str],
    ) -> None:
        details = []
        if last_status:
            details.append(f"最后状态={last_status}")
        if last_transient_error:
            details.append(f"最后查询错误={last_transient_error}")
        if last_response is not None:
            details.append(f"最后响应={_safe_json(last_response)}")
        detail_text = "；" + "；".join(details) if details else ""
        raise TimeoutError(
            f"抠图任务轮询超时：从成功取得 TaskId 起已达到 "
            f"{timeout_seconds:g} 秒；TaskId={task_id}；查询次数={query_count}"
            f"{detail_text}"
        )


def _get_token_from_environment() -> str:
    token = os.environ.get("BIMOAI_TOKEN", "").strip()
    if not token:
        raise BimoAISegmentError("缺少环境变量 BIMOAI_TOKEN。")
    return token


def _build_headers(token: str) -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Response-Format": "wrapped",
    }


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _validate_inputs(
    *,
    base_url: str,
    model_id: str,
    image_url: str,
    max_persons: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
    http_timeout_seconds: float,
) -> ValidatedInputs:
    base_url = _validate_base_url(base_url)
    model_id = _validate_model_id(model_id)
    image_url = _validate_image_url(image_url)
    max_persons = _validate_max_persons(max_persons)
    timeout_seconds = _positive_float(timeout_seconds, "timeout_seconds")
    http_timeout_seconds = _positive_float(http_timeout_seconds, "http_timeout_seconds")
    poll_interval_seconds = _positive_float(poll_interval_seconds, "poll_interval_seconds")

    return ValidatedInputs(
        base_url=base_url,
        model_id=model_id,
        image_url=image_url,
        max_persons=max_persons,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        http_timeout_seconds=http_timeout_seconds,
    )


def _validate_available_inputs(
    *,
    base_url: Optional[str],
    model_id: Optional[str],
    image_url: Optional[str],
    max_persons: Optional[int],
    poll_interval_seconds: Optional[float],
    timeout_seconds: Optional[float],
    http_timeout_seconds: Optional[float],
) -> None:
    """Validate prompt-time constants while ignoring unresolved node links."""
    if base_url is not None:
        _validate_base_url(base_url)
    if model_id is not None:
        _validate_model_id(model_id)
    if image_url is not None:
        _validate_image_url(image_url)
    if max_persons is not None:
        _validate_max_persons(max_persons)
    if poll_interval_seconds is not None:
        _positive_float(poll_interval_seconds, "poll_interval_seconds")
    if timeout_seconds is not None:
        _positive_float(timeout_seconds, "timeout_seconds")
    if http_timeout_seconds is not None:
        _positive_float(http_timeout_seconds, "http_timeout_seconds")


def _validate_base_url(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("base_url 必须是字符串。")

    base_url = value.strip().rstrip("/")
    parsed_base_url = urlparse(base_url)
    if parsed_base_url.scheme not in {"http", "https"} or not parsed_base_url.netloc:
        raise ValueError("base_url 必须是有效的 HTTP(S) 地址。")
    return base_url


def _validate_model_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("model_id 必须是字符串。")

    model_id = value.strip()
    if not model_id:
        raise ValueError("model_id 不能为空。")
    return model_id


def _validate_image_url(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("image_url 必须是字符串。")

    image_url = value.strip()
    parsed_image_url = urlparse(image_url)
    if parsed_image_url.scheme not in {"http", "https"} or not parsed_image_url.netloc:
        raise ValueError("image_url 必须是有效的 HTTP(S) URL。")

    if len(image_url) > 128:
        raise ValueError("image_url 不能超过接口规定的 128 个字符。")
    return image_url


def _validate_max_persons(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("max_persons 必须是整数，不能是布尔值。")

    try:
        max_persons = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("max_persons 必须是整数。") from exc

    if not 0 <= max_persons <= 6:
        raise ValueError("max_persons 必须在 0 到 6 之间。")
    return max_persons


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须是数字。") from exc

    if result <= 0:
        raise ValueError(f"{name} 必须大于 0。")

    return result


def _require_wrapped_data(payload: Dict[str, Any], *, context: str) -> Dict[str, Any]:
    data = payload.get("Data")
    if not isinstance(data, dict):
        msg = f"{context}的 wrapped 应答中，顶层 Data 不是对象；完整响应：{_safe_json(payload)}"
        raise BimoAISegmentError(msg)
    return data


def _require_nonempty_string(payload: Dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BimoAISegmentError(f"{context}缺少有效的 {key}；完整数据：{_safe_json(payload)}")
    return value.strip()


def _require_http_url(payload: Dict[str, Any], key: str, *, context: str) -> str:
    value = _require_nonempty_string(payload, key, context=context)
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BimoAISegmentError(f"{context}中的 {key} 不是有效的 HTTP(S) URL：{value}")
    return value


def _safe_json(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = repr(value)
    return _truncate(text, 3000)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


NODE_CLASS_MAPPINGS = {"BimoAISegmentImage": BimoAISegmentImage}
NODE_DISPLAY_NAME_MAPPINGS = {"BimoAISegmentImage": "BimoAI Image Segment"}
