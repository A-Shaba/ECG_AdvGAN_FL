// webapp/static/js/main.js

document.addEventListener("DOMContentLoaded", () => {
  console.log("Web dashboard loaded");

  const runButtons = document.querySelectorAll(".run-btn");
  const stepSelects = document.querySelectorAll(".step-select");
  const logBox = document.getElementById("logBox");
  const loader = document.getElementById("loader");
  const resetBtn = document.getElementById("resetBtn");

  // Handle each run button
  runButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const select = btn.previousElementSibling;
      const step = select.value;

      if (!step) {
        alert("No step selected");
        return;
      }

      // Disable everything
      toggleUI(false, "Running...");

      const formData = new FormData();
      formData.append("step", step);

      fetch("/run_step", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => {
          logBox.textContent = data.logs || "No output.";
        })
        .catch(err => {
          logBox.textContent = "Error: " + err;
        })
        .finally(() => {
          toggleUI(true);
        });
    });
  });

  // Handle reset button
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      if (!confirm("Are you sure you want to reset the project? This will delete all outputs, models, logs, etc.")) return;

      // Disable everything
      toggleUI(false, "Resetting...");

      fetch("/reset_project", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          logBox.textContent = data.logs || "Reset complete.";
        })
        .catch(err => {
          logBox.textContent = "Error: " + err;
        })
        .finally(() => {
          toggleUI(true);
        });
    });
  }

  // Helper to enable/disable UI
  function toggleUI(enable, loadingMessage = "") {
    runButtons.forEach(b => b.disabled = !enable);
    stepSelects.forEach(s => s.disabled = !enable);
    if (resetBtn) resetBtn.disabled = !enable;

    if (!enable) {
      loader.style.display = "block";
      logBox.textContent = loadingMessage;
    } else {
      loader.style.display = "none";
    }
  }
});
