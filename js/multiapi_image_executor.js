const { app } = window.comfyAPI.app;

const IMAGE_INPUT_LIMITS = {
  MAIE_GeminiTask: 6,
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

    const requested = Math.max(0, Math.min(maximum, Number(countWidget.value) || 0));
    const connected = numberedInputs(node, "image")
      .filter(({ input }) => input.link != null)
      .reduce((highest, { number }) => Math.max(highest, number), 0);
    const target = Math.max(requested, connected);

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
    if (nodeData.name !== "MAIE_ExecuteTasks" && !imageLimit) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      if (nodeData.name === "MAIE_ExecuteTasks") setupDynamicTaskInputs(this);
      if (imageLimit) setupDynamicImageInputs(this, imageLimit);
    };
  },
});
