// ============================================================================
// VeriSight AI - Frontend Logic
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneTitle = document.getElementById("dropzoneTitle");
  const dropzoneSubtitle = document.getElementById("dropzoneSubtitle");
  const samplesGrid = document.getElementById("samplesGrid");
  const modelSelect = document.getElementById("modelSelect");
  const modelAccBadge = document.getElementById("modelAccBadge");
  const colormapSelect = document.getElementById("colormapSelect");
  const thresholdRange = document.getElementById("thresholdRange");
  const thresholdVal = document.getElementById("thresholdVal");
  const btnAnalyze = document.getElementById("btnAnalyze");
  
  // Dashboard Elements
  const verdictBanner = document.getElementById("verdictBanner");
  const verdictText = document.getElementById("verdictText");
  const verdictSubtext = document.getElementById("verdictSubtext");
  const gaugeVal = document.getElementById("gaugeVal");
  const gaugeText = document.getElementById("gaugeText");
  
  const studioContainer = document.getElementById("studioContainer");
  const topLayer = document.getElementById("topLayer");
  const sliderHandle = document.getElementById("sliderHandle");
  const imgOriginal = document.getElementById("imgOriginal");
  const imgHeatmap = document.getElementById("imgHeatmap");
  const loadingOverlay = document.getElementById("loadingOverlay");
  
  const valTamperedArea = document.getElementById("valTamperedArea");
  const valIntensity = document.getElementById("valIntensity");
  const valLatency = document.getElementById("valLatency");
  const valDevice = document.getElementById("valDevice");
  
  // View Toggle Buttons
  const btnSplitView = document.getElementById("btnSplitView");
  const btnOverlayView = document.getElementById("btnOverlayView");
  const btnMaskView = document.getElementById("btnMaskView");

  // State
  let currentFile = null;
  let currentSamplePath = null;
  let lastPredictionData = null;
  let currentViewMode = "split"; // 'split', 'overlay', 'mask'
  let isDragging = false;

  // 1. Threshold range listener
  thresholdRange.addEventListener("input", (e) => {
    thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  // 2. Model selector accuracy badge update
  const modelAccMap = {
    "hybrid": "86.80% Acc (EXP 9)",
    "swin": "84.38% Acc (EXP 3)",
    "resnet50": "81.97% Acc (EXP 2)"
  };
  modelSelect.addEventListener("change", (e) => {
    modelAccBadge.textContent = modelAccMap[e.target.value] || "Active";
  });

  // 3. Fetch & Populate Samples
  async function loadSamples() {
    try {
      const res = await fetch("/api/samples");
      const samples = await res.json();
      samplesGrid.innerHTML = "";
      
      samples.forEach((sample, idx) => {
        const thumb = document.createElement("div");
        thumb.className = `sample-thumb ${idx === 0 ? "active" : ""}`;
        thumb.title = sample.name;
        
        const isFake = sample.type === "FAKE";
        thumb.innerHTML = `
          <img src="${sample.file}" alt="${sample.name}">
          <span class="sample-badge ${isFake ? 'fake' : 'real'}">${sample.category}</span>
        `;
        
        thumb.addEventListener("click", () => {
          document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
          thumb.classList.add("active");
          selectSample(sample.file, sample.name);
        });
        
        samplesGrid.appendChild(thumb);
      });

      // Automatically analyze the first sample on startup for instant demo
      if (samples.length > 0) {
        selectSample(samples[0].file, samples[0].name, true);
      }
    } catch (err) {
      console.warn("Error loading sample gallery:", err);
    }
  }

  function selectSample(filePath, name, autoRun = false) {
    currentFile = null;
    currentSamplePath = filePath;
    dropzoneTitle.textContent = name;
    dropzoneSubtitle.textContent = "Click to change selection or browse";
    
    // Set immediate preview
    imgOriginal.src = filePath;
    imgHeatmap.src = filePath;
    
    if (autoRun) {
      runInference();
    }
  }

  // 4. File Drag & Drop Handling
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    currentFile = file;
    currentSamplePath = null;
    document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("active"));
    
    dropzoneTitle.textContent = file.name;
    dropzoneSubtitle.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB • Ready for analysis`;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      imgOriginal.src = e.target.result;
      imgHeatmap.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  // 5. Run Inference Action
  btnAnalyze.addEventListener("click", runInference);

  async function runInference() {
    if (!currentFile && !currentSamplePath) {
      alert("Please upload an image or select a sample first!");
      return;
    }

    loadingOverlay.classList.add("active");
    btnAnalyze.disabled = true;

    const formData = new FormData();
    if (currentFile) {
      formData.append("image", currentFile);
    } else if (currentSamplePath) {
      formData.append("sample_path", currentSamplePath);
    }
    
    formData.append("model", modelSelect.value);
    formData.append("colormap", colormapSelect.value);
    formData.append("threshold", thresholdRange.value);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      if (data.error) {
        alert("Inference Error: " + data.error);
        return;
      }

      lastPredictionData = data;
      renderResults(data);
    } catch (err) {
      console.error("Analysis failed:", err);
      alert("Failed to analyze image. Please ensure the backend is running.");
    } finally {
      loadingOverlay.classList.remove("active");
      btnAnalyze.disabled = false;
    }
  }

  // 6. Render Results & Animations
  function renderResults(data) {
    const verdict = data.verdict;
    const loc = data.localization;
    const tele = data.telemetry;

    // A. Verdict Banner
    verdictBanner.className = `verdict-banner ${verdict.verdict_type || (verdict.is_fake ? 'fake' : 'real')}`;
    verdictText.textContent = verdict.label;
    verdictSubtext.textContent = verdict.notes || (verdict.is_fake 
      ? `Confidence: ${verdict.confidence}% • Probability: ${verdict.fake_probability}% Anomaly`
      : `Confidence: ${verdict.confidence}% • Probability: ${verdict.real_probability}% Real`);

    // B. Confidence Radial Gauge
    const circumference = 251.2;
    const offset = circumference - (verdict.confidence / 100) * circumference;
    gaugeVal.style.strokeDashoffset = offset;
    gaugeText.textContent = `${Math.round(verdict.confidence)}%`;

    // C. Images Setup
    imgOriginal.src = loc.original_image;
    updateStudioView(currentViewMode);

    // D. Telemetry Cards
    valTamperedArea.textContent = `${loc.tampered_area_pct}%`;
    valIntensity.textContent = `${loc.max_anomaly_intensity}%`;
    valLatency.textContent = `${tele.inference_time_ms} ms`;
    valDevice.textContent = `Backbone: ${tele.model_used.toUpperCase()}`;
  }

  function updateStudioView(mode) {
    if (!lastPredictionData) return;
    const loc = lastPredictionData.localization;
    
    currentViewMode = mode;
    [btnSplitView, btnOverlayView, btnMaskView].forEach(b => b.classList.remove("active"));
    
    if (mode === "split") {
      btnSplitView.classList.add("active");
      sliderHandle.style.display = "block";
      topLayer.style.display = "block";
      topLayer.style.width = "50%";
      sliderHandle.style.left = "50%";
      imgHeatmap.src = loc.overlay_image;
    } else if (mode === "overlay") {
      btnOverlayView.classList.add("active");
      sliderHandle.style.display = "none";
      topLayer.style.display = "none";
      imgHeatmap.src = loc.overlay_image;
    } else if (mode === "mask") {
      btnMaskView.classList.add("active");
      sliderHandle.style.display = "none";
      topLayer.style.display = "none";
      imgHeatmap.src = loc.binary_mask;
    }
  }

  btnSplitView.addEventListener("click", () => updateStudioView("split"));
  btnOverlayView.addEventListener("click", () => updateStudioView("overlay"));
  btnMaskView.addEventListener("click", () => updateStudioView("mask"));

  // 7. Interactive Split-Slider Interaction
  function setSliderPosition(x) {
    const rect = studioContainer.getBoundingClientRect();
    let pos = (x - rect.left) / rect.width;
    pos = Math.max(0.02, Math.min(0.98, pos));
    
    const pct = pos * 100;
    topLayer.style.width = `${pct}%`;
    sliderHandle.style.left = `${pct}%`;
  }

  studioContainer.addEventListener("mousedown", (e) => {
    if (currentViewMode !== "split") return;
    isDragging = true;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDragging = false;
  });

  // Touch support for mobile/tablets
  studioContainer.addEventListener("touchstart", (e) => {
    if (currentViewMode !== "split") return;
    isDragging = true;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    setSliderPosition(e.touches[0].clientX);
  });

  window.addEventListener("touchend", () => {
    isDragging = false;
  });

  // Initial call
  loadSamples();
});
