const { app } = window.comfyAPI.app;

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
    if (nodeData.name !== "MAIE_ExecuteTasks") return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      setupDynamicTaskInputs(this);
    };
  },
});
