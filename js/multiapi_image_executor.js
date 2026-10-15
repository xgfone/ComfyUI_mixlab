const { app } = window.comfyAPI.app;

const IMAGE_INPUT_LIMITS = {
  MAIE_GeminiTask: 14,
  MAIE_SeedreamTask: 10,
  MAIE_GPTImageTask: 16,
  MAIE_QwenTask: 3,
};

function numberedInputs(node, prefix) {
  return (node.inputs ?? [])
    .map((input, index) => ({
      input,
      index,
      match: new RegExp(`^${prefix}(\\d+)$`).exec(input.name),
    }))
    .filter(({ match }) => match)
    .map(({ input, index, match }) => ({ input, index, number: Number(match[1]) }));
}

function setupDynamicImageInputs(node, maximum) {
  const rebuild = () => {
    const countWidget = node.widgets?.find((widget) => widget.name === "imagecount");
    if (!countWidget) return;

    const limit = typeof maximum === "function" ? maximum() : maximum;
    countWidget.options = { ...countWidget.options, max: limit };
    const requested = Math.max(0, Math.min(limit, Math.floor(Number(countWidget.value) || 0)));
    const connected = numberedInputs(node, "image")
      .filter(({ input }) => input.link != null)
      .reduce((highest, { number }) => Math.max(highest, number), 0);
    const target = Math.max(requested, connected);
    countWidget.value = target;

    for (const { index, number, input } of numberedInputs(node, "image").sort(
      (a, b) => b.index - a.index,
    )) {
      if (number > target && input.link == null) node.removeInput(index);
    }
    const existing = new Set(numberedInputs(node, "image").map(({ number }) => number));
    for (let number = 1; number <= target; number += 1) {
      if (!existing.has(number)) node.addInput(`image${number}`, "IMAGE", { shape: 7 });
    }

    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
  };

  node._maieRebuildImageInputs = rebuild;

  node.addWidget("button", "更新图片输入", null, rebuild);
  const countWidget = node.widgets?.find((widget) => widget.name === "imagecount");
  if (countWidget) {
    const original = countWidget.callback;
    countWidget.callback = function (value, canvas) {
      const result = original?.apply(this, arguments);
      if (!canvas) rebuild();
      return result;
    };
  }
  rebuild();
}

function parseSmartConfig(value) {
  const punctuation = { "，": ",", "：": ":", "｛": "{", "｝": "}", "［": "[", "］": "]" };
  const quotes = { '"': '"', "“": "”", "＂": "＂", "‘": "’" };
  let output = "",
    closing = null,
    escaped = false;
  for (const character of String(value ?? "").replace(/^\uFEFF/, "")) {
    if (character === "\r" || character === "\n") {
      if (closing == null) output += " ";
      continue;
    }
    if (closing != null) {
      if (escaped) {
        output += character;
        escaped = false;
      } else if (character === "\\") {
        output += character;
        escaped = true;
      } else if (character === closing) {
        output += '"';
        closing = null;
      } else if (character === '"') output += '\\"';
      else if (character === "\t") output += "\\t";
      else output += character;
    } else if (quotes[character]) {
      closing = quotes[character];
      output += '"';
    } else output += punctuation[character] ?? character;
  }
  return output.trim() ? JSON.parse(output) : {};
}

function setupSmartInputs(node, nodeData) {
  const profiles = nodeData.input.required.model[1].smart_profiles;
  const widget = (name) => node.widgets?.find((item) => item.name === name);
  const effectiveModel = () => {
    try {
      return parseSmartConfig(widget("override_config")?.value).model ?? widget("model")?.value;
    } catch {
      return widget("model")?.value; // The backend reports malformed JSON at execution.
    }
  };
  const profile = () =>
    Object.entries(profiles).find(([, value]) => String(effectiveModel()).startsWith(value.prefix));
  const maximum = () => {
    // Connected configuration/model values are resolved only during execution.
    const unresolved = node.inputs?.some(
      (input) => ["override_config", "model"].includes(input.name) && input.link != null,
    );
    return unresolved
      ? Math.max(...Object.values(profiles).map((value) => value.image_limit))
      : (profile()?.[1].image_limit ?? 16);
  };
  const visibleState = new WeakMap();
  const update = () => {
    const selected = profile();
    const supported = new Set(
      selected?.[1].parameters ?? Object.values(profiles).flatMap((value) => value.parameters),
    );
    if (selected?.[0] === "seedream") {
      const model = effectiveModel();
      if (
        [
          "doubao-seedream-5-0-pro-260628",
          "doubao-seedream-5-0-pro",
          "doubao-seedream-5-0-flash-260915",
        ].includes(model)
      ) {
        supported.delete("sequential_image_generation");
        if (model === "doubao-seedream-5-0-flash-260915")
          supported.delete("optimize_prompt_options");
      } else supported.delete("optimize_prompt_options");
    }
    for (const item of node.widgets ?? []) {
      if (
        ["prompt", "model", "override_config", "imagecount"].includes(item.name) ||
        item.type === "button"
      )
        continue;
      if (!visibleState.has(item))
        visibleState.set(item, { type: item.type, computeSize: item.computeSize });
      const original = visibleState.get(item);
      const visible = supported.has(item.name);
      item.type = visible ? original.type : "converted-widget";
      item.computeSize = visible ? original.computeSize : () => [0, -4];
      item.hidden = !visible;
      if (item.element?.style) item.element.style.display = visible ? "" : "none";
    }
    node._maieRebuildImageInputs?.();
  };
  setupDynamicImageInputs(node, maximum);
  for (const name of ["model", "override_config"]) {
    const control = widget(name);
    if (!control) continue;
    const original = control.callback;
    control.callback = function () {
      const result = original?.apply(this, arguments);
      update();
      return result;
    };
  }
  for (const event of ["onConfigure", "onConnectionsChange"]) {
    const original = node[event];
    node[event] = function () {
      const result = original?.apply(this, arguments);
      update();
      return result;
    };
  }
  update();
}

function setupDynamicTaskInputs(node) {
  const rebuild = () => {
    const countWidget = node.widgets?.find((widget) => widget.name === "inputcount");
    if (!countWidget) return;
    const target = Math.max(1, Number(countWidget.value) || 1);
    const taskInputs = () => node.inputs?.filter((input) => /^task_\d+$/.test(input.name)) ?? [];
    let current = taskInputs().length;

    while (current > target) {
      const last = node.inputs.map((input) => input.name).lastIndexOf(`task_${current}`);
      if (last >= 0) node.removeInput(last);
      current -= 1;
    }
    while (current < target) {
      current += 1;
      node.addInput(`task_${current}`, "MULTIAPI_IMAGE_TASK", { shape: 7 });
    }
    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
  };

  node.addWidget("button", "更新输入", null, rebuild);
  const countWidget = node.widgets?.find((widget) => widget.name === "inputcount");
  if (countWidget) {
    const original = countWidget.callback;
    countWidget.callback = function (value, canvas) {
      const result = original?.apply(this, arguments);
      if (!canvas) rebuild();
      return result;
    };
  }
}

app.registerExtension({
  name: "multiapi_image_executor.dynamic_tasks",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const imageLimit = IMAGE_INPUT_LIMITS[nodeData.name];
    if (nodeData.name !== "MAIE_ExecuteTasks" && nodeData.name !== "MAIE_SmartTask" && !imageLimit)
      return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      if (nodeData.name === "MAIE_ExecuteTasks") setupDynamicTaskInputs(this);
      if (nodeData.name === "MAIE_SmartTask") setupSmartInputs(this, nodeData);
      if (imageLimit) setupDynamicImageInputs(this, imageLimit);
    };
  },
});
