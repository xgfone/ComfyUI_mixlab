import { app } from "../../scripts/app.js";

const NODE_CLASS = "BIMO_CornerPinPerspective";
const CORNERS = [
  { id: "tl", label: "TL", x: "tl_x", y: "tl_y" },
  { id: "tr", label: "TR", x: "tr_x", y: "tr_y" },
  { id: "br", label: "BR", x: "br_x", y: "br_y" },
  { id: "bl", label: "BL", x: "bl_x", y: "bl_y" },
];

function findWidget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function clamp(v, lo, hi) {
  v = Number(v);
  if (!Number.isFinite(v)) return 0;
  return Math.max(lo, Math.min(hi, v));
}

function getCoordRange(node) {
  const editor = node.__bimoCornerEditor;
  if (!editor) return { minX: -0.5, maxX: 1.5, minY: -0.5, maxY: 1.5 };
  return editor.coordRange;
}

function getWidgetNumber(node, name, fallback = 0) {
  const w = findWidget(node, name);
  if (!w) return fallback;
  const v = Number(w.value);
  return Number.isFinite(v) ? v : fallback;
}

function setWidgetNumber(node, name, value) {
  const w = findWidget(node, name);
  if (!w) return;
  const next = Math.round(Number(value) * 10000) / 10000;
  if (!Number.isFinite(next) || w.value === next) return;
  w.value = next;
  if (typeof w.callback === "function") {
    w.callback(next, app.canvas, node, app.canvas?.graph_mouse);
  }
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function isBoolWidgetOn(node, name) {
  const w = findWidget(node, name);
  return w?.value === true || w?.value === "true" || w?.value === 1 || w?.value === "1";
}

function getPoints(node) {
  return CORNERS.map((c) => ({
    ...c,
    nx: getWidgetNumber(node, c.x, c.id === "tr" || c.id === "br" ? 1 : 0),
    ny: getWidgetNumber(node, c.y, c.id === "br" || c.id === "bl" ? 1 : 0),
  }));
}

function getNaturalAspect(node) {
  const img = node.imgs?.[0];
  if (img?.naturalWidth > 0 && img?.naturalHeight > 0) return img.naturalWidth / img.naturalHeight;
  if (img?.width > 0 && img?.height > 0) return img.width / img.height;
  return 1;
}

function fitRectKeepAspect(x, y, w, h, aspect) {
  let rw = w;
  let rh = w / aspect;
  if (rh > h) {
    rh = h;
    rw = h * aspect;
  }
  return { x: x + (w - rw) / 2, y: y + (h - rh) / 2, w: rw, h: rh };
}

function distanceSquared(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

function computeCoordRange(points) {
  let minX = Math.min(0, ...points.map((p) => p.nx));
  let maxX = Math.max(1, ...points.map((p) => p.nx));
  let minY = Math.min(0, ...points.map((p) => p.ny));
  let maxY = Math.max(1, ...points.map((p) => p.ny));

  const padX = Math.max(0.25, (maxX - minX) * 0.18);
  const padY = Math.max(0.25, (maxY - minY) * 0.18);
  minX -= padX; maxX += padX;
  minY -= padY; maxY += padY;

  // Keep the editor usable and avoid runaway dragging. Backend still accepts
  // the numeric widgets up to their widget min/max.
  minX = Math.max(-4, minX); maxX = Math.min(5, maxX);
  minY = Math.max(-4, minY); maxY = Math.min(5, maxY);
  if (maxX - minX < 0.01) maxX = minX + 0.01;
  if (maxY - minY < 0.01) maxY = minY + 0.01;
  return { minX, maxX, minY, maxY };
}

function coordToPixel(p, rect, range) {
  return {
    x: rect.x + ((p.nx - range.minX) / (range.maxX - range.minX)) * rect.w,
    y: rect.y + ((p.ny - range.minY) / (range.maxY - range.minY)) * rect.h,
  };
}

function pixelToCoord(px, py, rect, range) {
  return {
    nx: range.minX + ((px - rect.x) / rect.w) * (range.maxX - range.minX),
    ny: range.minY + ((py - rect.y) / rect.h) * (range.maxY - range.minY),
  };
}

function drawChecker(ctx, x, y, w, h, cell = 12) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  for (let yy = y; yy < y + h; yy += cell) {
    for (let xx = x; xx < x + w; xx += cell) {
      const odd = (Math.floor((xx - x) / cell) + Math.floor((yy - y) / cell)) % 2;
      ctx.fillStyle = odd ? "rgba(255,255,255,0.13)" : "rgba(255,255,255,0.06)";
      ctx.fillRect(xx, yy, cell, cell);
    }
  }
  ctx.restore();
}

// Canvas 2D supports affine triangle mapping, not true perspective mapping.
// Splitting into two triangles gives a fast, useful realtime preview. The
// backend still performs the final high-quality perspective transform.
function drawImageTriangle(ctx, img, sx0, sy0, sx1, sy1, sx2, sy2, dx0, dy0, dx1, dy1, dx2, dy2) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(dx0, dy0);
  ctx.lineTo(dx1, dy1);
  ctx.lineTo(dx2, dy2);
  ctx.closePath();
  ctx.clip();

  const denom = sx0 * (sy1 - sy2) + sx1 * (sy2 - sy0) + sx2 * (sy0 - sy1);
  if (Math.abs(denom) < 1e-6) {
    ctx.restore();
    return;
  }

  const a = (dx0 * (sy1 - sy2) + dx1 * (sy2 - sy0) + dx2 * (sy0 - sy1)) / denom;
  const b = (dy0 * (sy1 - sy2) + dy1 * (sy2 - sy0) + dy2 * (sy0 - sy1)) / denom;
  const c = (dx0 * (sx2 - sx1) + dx1 * (sx0 - sx2) + dx2 * (sx1 - sx0)) / denom;
  const d = (dy0 * (sx2 - sx1) + dy1 * (sx0 - sx2) + dy2 * (sx1 - sx0)) / denom;
  const e = (dx0 * (sx1 * sy2 - sx2 * sy1) + dx1 * (sx2 * sy0 - sx0 * sy2) + dx2 * (sx0 * sy1 - sx1 * sy0)) / denom;
  const f = (dy0 * (sx1 * sy2 - sx2 * sy1) + dy1 * (sx2 * sy0 - sx0 * sy2) + dy2 * (sx0 * sy1 - sx1 * sy0)) / denom;

  ctx.transform(a, b, c, d, e, f);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}

function drawWarpedPreview(ctx, node, rect, ptsPx) {
  drawChecker(ctx, rect.x, rect.y, rect.w, rect.h, 12);
  const img = node.imgs?.[0];

  if (img && img.complete !== false && (img.naturalWidth || img.width) > 0) {
    const iw = img.naturalWidth || img.width;
    const ih = img.naturalHeight || img.height;
    const p = ptsPx;
    try {
      drawImageTriangle(ctx, img, 0, 0, iw, 0, iw, ih, p[0].x, p[0].y, p[1].x, p[1].y, p[2].x, p[2].y);
      drawImageTriangle(ctx, img, 0, 0, iw, ih, 0, ih, p[0].x, p[0].y, p[2].x, p[2].y, p[3].x, p[3].y);
    } catch (e) {
      // Fall back to outline preview.
    }
  } else {
    ctx.save();
    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(255,255,255,0.62)";
    ctx.fillText("Run once to load image preview", rect.x + rect.w / 2, rect.y + rect.h / 2);
    ctx.restore();
  }
}

function installGlobalDragHandlers(node, state) {
  if (state.globalHandlersInstalled) return;
  state.globalHandlersInstalled = true;

  state.onPointerMove = (event) => {
    if (!state.dragging || !state.rect || !isBoolWidgetOn(node, "edit_enabled")) return;
    event.preventDefault?.();
    event.stopPropagation?.();
    event.stopImmediatePropagation?.();

    const canvas = app.canvas?.canvas;
    const scale = app.canvas?.ds?.scale || 1;
    const offset = app.canvas?.ds?.offset || [0, 0];
    const bounds = canvas?.getBoundingClientRect?.();
    if (!bounds) return;

    // Convert browser pixels to graph coordinates.
    const graphX = (event.clientX - bounds.left) / scale - offset[0];
    const graphY = (event.clientY - bounds.top) / scale - offset[1];
    const localX = graphX - node.pos[0];
    const localY = graphY - node.pos[1];
    const c = pixelToCoord(localX, localY, state.rect, state.coordRange);
    const corner = CORNERS.find((it) => it.id === state.dragging);
    if (!corner) return;

    const nx = clamp(c.nx, -4, 5);
    const ny = clamp(c.ny, -4, 5);
    setWidgetNumber(node, corner.x, nx);
    setWidgetNumber(node, corner.y, ny);
  };

  state.onPointerUp = (event) => {
    if (!state.dragging) return;
    event.preventDefault?.();
    event.stopPropagation?.();
    event.stopImmediatePropagation?.();
    state.dragging = null;
    node.flags = node.flags || {};
    node.flags.pinned = state.wasPinned || false;
    app.canvas.node_dragged = null;
    app.canvas.selected_group = null;
    node.setDirtyCanvas?.(true, true);
  };

  document.addEventListener("pointermove", state.onPointerMove, true);
  document.addEventListener("pointerup", state.onPointerUp, true);
  document.addEventListener("mouseup", state.onPointerUp, true);
}

function createCornerEditorWidget(node) {
  const state = {
    dragging: null,
    rect: null,
    coordRange: { minX: -0.5, maxX: 1.5, minY: -0.5, maxY: 1.5 },
    wasPinned: false,
    globalHandlersInstalled: false,
  };
  node.__bimoCornerEditor = state;

  const widget = {
    name: "corner_pin_editor",
    type: "custom",
    value: null,

    computeSize(width) {
      const aspect = getNaturalAspect(node);
      const h = Math.max(210, Math.min(520, width / Math.max(0.35, Math.min(2.5, aspect)) + 76));
      return [width, h];
    },

    draw(ctx, node, width, y, height) {
      const enabled = isBoolWidgetOn(node, "edit_enabled");
      const expand = isBoolWidgetOn(node, "expand_canvas");
      const pad = 14;
      const headerH = 26;
      const footerH = 20;
      const aspect = getNaturalAspect(node);

      const viewport = fitRectKeepAspect(
        pad,
        y + headerH + pad,
        Math.max(60, width - pad * 2),
        Math.max(60, height - headerH - footerH - pad * 2),
        aspect
      );

      const pts = getPoints(node);
      state.coordRange = computeCoordRange(pts);
      state.rect = viewport;

      const ptsPx = pts.map((p) => ({ ...p, ...coordToPixel(p, viewport, state.coordRange) }));

      ctx.save();
      ctx.font = "12px sans-serif";
      ctx.textBaseline = "middle";
      ctx.fillStyle = enabled ? "#ffffff" : "#b8b8b8";
      ctx.fillText(enabled ? "Corner Editor: ON - drag handles" : "Corner Editor: LOCKED", pad, y + 12);
      ctx.fillStyle = expand ? "rgba(100,210,255,0.9)" : "rgba(255,255,255,0.55)";
      ctx.textAlign = "right";
      ctx.fillText(expand ? "Expand Canvas" : "Crop to Source Size", width - pad, y + 12);
      ctx.textAlign = "left";

      // Draw original-aspect viewport and warped preview.
      ctx.fillStyle = "rgba(16,16,16,0.80)";
      ctx.fillRect(viewport.x, viewport.y, viewport.w, viewport.h);
      drawWarpedPreview(ctx, node, viewport, ptsPx);

      // Original normalized image bounds in current coordinate range.
      const tl = coordToPixel({ nx: 0, ny: 0 }, viewport, state.coordRange);
      const br = coordToPixel({ nx: 1, ny: 1 }, viewport, state.coordRange);
      ctx.strokeStyle = "rgba(255,255,255,0.32)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);
      ctx.setLineDash([]);

      // Polygon outline.
      ctx.beginPath();
      ctx.moveTo(ptsPx[0].x, ptsPx[0].y);
      for (let i = 1; i < ptsPx.length; i++) ctx.lineTo(ptsPx[i].x, ptsPx[i].y);
      ctx.closePath();
      ctx.strokeStyle = enabled ? "rgba(80,190,255,1)" : "rgba(210,210,210,0.65)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Handles.
      for (const p of ptsPx) {
        const r = state.dragging === p.id ? 8 : 6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = enabled ? "#ffffff" : "#909090";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = enabled ? "rgba(80,190,255,1)" : "rgba(150,150,150,1)";
        ctx.stroke();
        ctx.font = "11px sans-serif";
        ctx.fillStyle = enabled ? "#ffffff" : "#b8b8b8";
        ctx.fillText(p.label, p.x + 8, p.y - 8);
      }

      // Footer with current coordinate window.
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      const rangeText = `view x:${state.coordRange.minX.toFixed(2)}..${state.coordRange.maxX.toFixed(2)} y:${state.coordRange.minY.toFixed(2)}..${state.coordRange.maxY.toFixed(2)}`;
      ctx.fillText(rangeText, pad, y + height - 10);
      ctx.restore();
    },

    mouse(event, pos, node) {
      if (!state.rect) return false;
      const enabled = isBoolWidgetOn(node, "edit_enabled");
      if (!enabled) {
        state.dragging = null;
        return false;
      }

      const type = event.type;
      const isDown = type === "mousedown" || type === "pointerdown";
      const isMove = type === "mousemove" || type === "pointermove";
      const isUp = type === "mouseup" || type === "pointerup" || type === "mouseleave";

      const px = pos?.[0] ?? 0;
      const py = pos?.[1] ?? 0;
      const rect = state.rect;
      const ptsPx = getPoints(node).map((p) => ({ ...p, ...coordToPixel(p, rect, state.coordRange) }));

      if (isDown) {
        let best = null;
        let bestD = Infinity;
        const radius = 20;
        for (const p of ptsPx) {
          const d = distanceSquared(px, py, p.x, p.y);
          if (d < bestD) {
            bestD = d;
            best = p;
          }
        }

        if (best && bestD <= radius * radius) {
          state.dragging = best.id;
          state.wasPinned = node.flags?.pinned || false;
          node.flags = node.flags || {};
          node.flags.pinned = true;
          app.canvas.node_dragged = null;
          installGlobalDragHandlers(node, state);
          event.preventDefault?.();
          event.stopPropagation?.();
          event.stopImmediatePropagation?.();
          return true;
        }
      }

      if (isMove && state.dragging) {
        const c = pixelToCoord(px, py, rect, state.coordRange);
        const corner = CORNERS.find((it) => it.id === state.dragging);
        if (corner) {
          setWidgetNumber(node, corner.x, clamp(c.nx, -4, 5));
          setWidgetNumber(node, corner.y, clamp(c.ny, -4, 5));
        }
        event.preventDefault?.();
        event.stopPropagation?.();
        event.stopImmediatePropagation?.();
        return true;
      }

      if (isUp && state.dragging) {
        state.dragging = null;
        node.flags = node.flags || {};
        node.flags.pinned = state.wasPinned || false;
        event.preventDefault?.();
        event.stopPropagation?.();
        event.stopImmediatePropagation?.();
        return true;
      }

      return false;
    },
  };

  node.addCustomWidget(widget);
  node.size = node.size || [420, 720];
  node.size[0] = Math.max(node.size[0], 420);
  node.size[1] = Math.max(node.size[1], 720);
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "BIMO.CornerPinEditor.v5",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_CLASS) return;
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      const exists = this.widgets?.some((w) => w.name === "corner_pin_editor");
      if (!exists) createCornerEditorWidget(this);
      return result;
    };
  },
});
