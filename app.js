/* ==========================================================================
   AuraSeg - Premium Customer Segmentation JS Logic
   ========================================================================== */

// --- Constants & Configuration ---
const SCALER_MEAN = [60.56, 50.2];
const SCALER_SCALE = [26.19897708, 25.75888196];

const CLUSTER_CENTROIDS = [
  [-0.20091257, -0.02645617], // Cluster 0
  [0.99158305, 1.23950275],   // Cluster 1
  [-1.32954532, 1.13217788],  // Cluster 2
  [1.05500302, -1.28443907],  // Cluster 3
  [-1.30751869, -1.13696536]  // Cluster 4
];

const SEGMENTS = {
  0: {
    name: "Middle of the Road",
    badgeClass: "c-0",
    color: "#a855f7",
    bgColor: "rgba(168, 85, 247, 0.15)",
    desc: "Moderate annual income and moderate spending habits. They represent the largest customer demographic group and are generally stable and brand-neutral.",
    avgIncome: "₹45.9L",
    avgSpending: "49.5",
    strategy: "Engage with classic loyalty programs, milestone-based reward coupons, and mid-tier product recommendations."
  },
  1: {
    name: "High-Value Targets (Star)",
    badgeClass: "c-1",
    color: "#10b981",
    bgColor: "rgba(16, 185, 129, 0.15)",
    desc: "High annual income paired with high spending score. Mostly young to middle-aged adults seeking premium boutique items, exclusive rewards, and direct service.",
    avgIncome: "₹71.8L",
    avgSpending: "82.1",
    strategy: "Target with early access to luxury collections, invite-only VIP events, customized high-end styling, and premium product lines."
  },
  2: {
    name: "Trendsetting Spendthrifts",
    badgeClass: "c-2",
    color: "#3b82f6",
    bgColor: "rgba(59, 130, 246, 0.15)",
    desc: "Low annual income but very high spending score. Primarily composed of younger consumers who prioritize modern fashion trends, social proof, and buy-now-pay-later financing.",
    avgIncome: "₹21.3L",
    avgSpending: "79.4",
    strategy: "Deploy dynamic social media marketing, short-term flash discounts, trendy budget accessories, and seamless digital checkout options."
  },
  3: {
    name: "Careful Skeptics",
    badgeClass: "c-3",
    color: "#f59e0b",
    bgColor: "rgba(245, 158, 11, 0.15)",
    desc: "High annual income combined with low spending score. Methodical shoppers who focus on quality, return on investment, and product utility. Mostly male demographically.",
    avgIncome: "₹73.2L",
    avgSpending: "17.1",
    strategy: "Focus on durability, utility, detailed feature lists, and long-term value. Offer trial periods, extensions on warranties, and high-quality customer service."
  },
  4: {
    name: "Frugal Conservatives",
    badgeClass: "c-4",
    color: "#ef4444",
    bgColor: "rgba(239, 68, 68, 0.15)",
    desc: "Low income matching low spending scores. Older customer demographic who prioritize essentials, budget-friendliness, and practical products.",
    avgIncome: "₹21.8L",
    avgSpending: "20.9",
    strategy: "Offer everyday lowest-price guarantees, bulk discounts, essential product categories, and simple, zero-friction loyalty programs."
  }
};

// --- Application State ---
let dbCustomers = [];       // Loaded 200 customers
let filteredCustomers = []; // Filtered customers for search/filter table
let currentPage = 1;
const rowsPerPage = 10;
let sortColumn = 0;         // 0: ID, 1: Gender, 2: Age, 3: Income, 4: Spending, 5: Cluster
let sortDirection = 1;      // 1: Ascending, -1: Descending

let scatterChart = null;
let showCentroids = true;

// Batch upload state
let uploadedCSVData = [];
let batchChart = null;

// --- DOM References ---
const ageSlider = document.getElementById("input-age");
const incomeSlider = document.getElementById("input-income");
const spendingSlider = document.getElementById("input-spending");

const ageVal = document.getElementById("age-val");
const incomeVal = document.getElementById("income-val");
const spendingVal = document.getElementById("spending-val");

const resBadge = document.getElementById("res-badge");
const resTitle = document.getElementById("res-title");
const resDesc = document.getElementById("res-desc");
const resIncome = document.getElementById("res-income-avg");
const resSpending = document.getElementById("res-spending-avg");
const resStrategy = document.getElementById("res-strategy");
const predictionOutput = document.getElementById("prediction-result");

// --- Currency Formatter Helpers (Indian Standard Rupee) ---
function formatRupeeFull(incomeInK) {
  const value = incomeInK * 83000;
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0
  }).format(value);
}

function formatRupeeCompact(incomeInK) {
  const value = incomeInK * 83000;
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    notation: 'compact',
    maximumFractionDigits: 1
  }).format(value);
}

function getIncomeColumnName(row) {
  const keys = Object.keys(row);
  const incomeKeys = [
    "Annual Income (k$)",
    "Annual Income (k₹)",
    "Annual Income (Lakhs)",
    "Annual Income (INR)",
    "Annual Income"
  ];
  return keys.find(k => incomeKeys.includes(k));
}

function normalizeIncomeToKUsd(val, columnName) {
  if (!val || isNaN(val)) return 60; // default mean
  val = parseFloat(val);
  if (val > 1000) {
    // It's raw INR (e.g. 1,245,000)
    return val / 83000;
  }
  if (columnName && columnName.toLowerCase().includes("lakh")) {
    // It's in Lakhs (e.g. 12.45)
    return val / 0.83;
  }
  // It's in the base k$ / index scale (e.g. 15 to 137)
  return val;
}

// --- Core Machine Learning Logic (Scaler & KMeans Predict) ---
function scaleInput(income, spending) {
  const scaledIncome = (income - SCALER_MEAN[0]) / SCALER_SCALE[0];
  const scaledSpending = (spending - SCALER_MEAN[1]) / SCALER_SCALE[1];
  return [scaledIncome, scaledSpending];
}

function unscaleInput(scaledIncome, scaledSpending) {
  const income = (scaledIncome * SCALER_SCALE[0]) + SCALER_MEAN[0];
  const spending = (scaledSpending * SCALER_SCALE[1]) + SCALER_MEAN[1];
  return [income, spending];
}

function predictCluster(income, spending) {
  const [x, y] = scaleInput(income, spending);
  let minDistance = Infinity;
  let predictedCluster = 0;

  for (let i = 0; i < CLUSTER_CENTROIDS.length; i++) {
    const cx = CLUSTER_CENTROIDS[i][0];
    const cy = CLUSTER_CENTROIDS[i][1];
    // Euclidean distance squared
    const dist = Math.pow(x - cx, 2) + Math.pow(y - cy, 2);
    if (dist < minDistance) {
      minDistance = dist;
      predictedCluster = i;
    }
  }
  return predictedCluster;
}

// --- Initialize App ---
document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide Icons
  lucide.createIcons();
  
  // Attach event listeners to sliders
  ageSlider.addEventListener("input", updatePrediction);
  incomeSlider.addEventListener("input", updatePrediction);
  spendingSlider.addEventListener("input", updatePrediction);
  
  // Initialize Gender Toggles to update visual state/prediction
  document.querySelectorAll('input[name="gender"]').forEach(el => {
    el.addEventListener("change", updatePrediction);
  });

  // Setup drag-and-drop
  setupDragAndDrop();

  // Load the initial 200 customers dataset
  loadBaseDataset();
});

// --- Tab Swapping ---
window.switchTab = function(tabId) {
  // Update buttons active class
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.getElementById(`tab-${tabId}-btn`).classList.add("active");

  // Show proper viewport
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  document.getElementById(`pane-${tabId}`).classList.add("active");

  // Redraw charts if necessary to fix scaling
  if (tabId === 'visualizer' && scatterChart) {
    scatterChart.resize();
  }
  if (tabId === 'batch' && batchChart) {
    batchChart.resize();
  }
};

// --- Live Prediction Updates ---
function updatePrediction() {
  const age = parseInt(ageSlider.value);
  const income = parseInt(incomeSlider.value);
  const spending = parseInt(spendingSlider.value);

  // Update slider label values
  ageVal.textContent = age;
  incomeVal.textContent = formatRupeeCompact(income);
  spendingVal.textContent = spending;

  // Run model prediction
  const cluster = predictCluster(income, spending);
  const segment = SEGMENTS[cluster];

  // Update UI Prediction Cards
  resBadge.textContent = `Segment ${cluster}`;
  resBadge.style.color = segment.color;
  resBadge.style.backgroundColor = segment.bgColor;
  
  resTitle.textContent = segment.name;
  resDesc.textContent = segment.desc;
  resIncome.textContent = `Avg: ${segment.avgIncome}`;
  resSpending.textContent = `Avg: ${segment.avgSpending}`;
  resStrategy.textContent = segment.strategy;

  // Update Sidebar style class (border effect)
  predictionOutput.className = `prediction-output c-${cluster}`;

  // Update the glowing cursor point on the Chart
  if (scatterChart) {
    const liveDatasetIndex = scatterChart.data.datasets.findIndex(d => d.label === 'Active Profile');
    if (liveDatasetIndex !== -1) {
      scatterChart.data.datasets[liveDatasetIndex].data = [{ x: income, y: spending }];
      
      // Update color matching predicted segment
      scatterChart.data.datasets[liveDatasetIndex].pointBackgroundColor = segment.color;
      scatterChart.data.datasets[liveDatasetIndex].pointBorderColor = '#ffffff';
      
      scatterChart.update('none'); // Update without full animation for performance
    }
  }
}

// --- Fetch & Parse Dataset ---
function loadBaseDataset() {
  fetch("Mall_Customers.csv")
    .then(res => {
      if (!res.ok) throw new Error("Dataset file not found or load failed.");
      return res.text();
    })
    .then(text => {
      const parsed = parseCSV(text);
      
      // Map to proper types and calculate cluster labels
      dbCustomers = parsed.map(c => {
        const id = parseInt(c["CustomerID"]);
        const gender = c["Gender"];
        const age = parseInt(c["Age"]);
        const incomeKey = getIncomeColumnName(c);
        const income = normalizeIncomeToKUsd(c[incomeKey], incomeKey);
        const spending = parseInt(c["Spending Score (1-100)"]);
        const cluster = predictCluster(income, spending);
        return { id, gender, age, income, spending, cluster };
      });
      
      filteredCustomers = [...dbCustomers];
      
      // Initialize interactive scatter chart
      initScatterChart();
      
      // Initialize database table view
      renderDatabaseTable();
      
      // Trigger initial prediction update to align chart points
      updatePrediction();
    })
    .catch(err => {
      console.error("Failed to load Mall Customers dataset:", err);
      // Fail gracefully: insert dummy data or notify the user
      alert("Note: Mall_Customers.csv failed to load from workspace root. Batch features and dataset table may be limited. Loading placeholder points...");
    });
}

function parseCSV(text) {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",").map(h => h.trim());
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    if (!lines[i].trim()) continue;
    
    // Split by comma, handling potential quotes (standard CSV)
    const values = [];
    let currentVal = "";
    let inQuotes = false;
    for (let char of lines[i]) {
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        values.push(currentVal.trim());
        currentVal = "";
      } else {
        currentVal += char;
      }
    }
    values.push(currentVal.trim());

    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] || "";
    });
    data.push(row);
  }
  return data;
}

// --- Interactive Scatter Chart (Chart.js) ---
function initScatterChart() {
  const ctx = document.getElementById("segmentationChart").getContext("2d");
  
  // Group 200 customers by cluster for datasets
  const clusterDatasets = [];
  for (let k = 0; k < 5; k++) {
    const clusterPoints = dbCustomers
      .filter(c => c.cluster === k)
      .map(c => ({ x: c.income, y: c.spending, id: c.id, age: c.age, gender: c.gender }));
      
    clusterDatasets.push({
      label: `Segment ${k} (${SEGMENTS[k].name})`,
      data: clusterPoints,
      backgroundColor: SEGMENTS[k].bgColor,
      borderColor: SEGMENTS[k].color,
      borderWidth: 1,
      pointRadius: 6,
      pointHoverRadius: 8,
      pointHoverBackgroundColor: SEGMENTS[k].color,
      pointHoverBorderColor: '#ffffff',
      pointHoverBorderWidth: 2
    });
  }

  // Add Centroids dataset
  const centroidsData = CLUSTER_CENTROIDS.map((c, index) => {
    const [income, spending] = unscaleInput(c[0], c[1]);
    return { x: income, y: spending, label: `Centroid ${index}` };
  });

  clusterDatasets.push({
    label: 'Centroids',
    data: centroidsData,
    backgroundColor: '#ef4444',
    borderColor: '#ffffff',
    borderWidth: 2,
    pointStyle: 'crossRot',
    pointRadius: 12,
    pointHoverRadius: 14,
    hidden: !showCentroids
  });

  // Add Real-time preview node
  clusterDatasets.push({
    label: 'Active Profile',
    data: [{ x: 60, y: 50 }],
    backgroundColor: SEGMENTS[0].color,
    borderColor: '#ffffff',
    borderWidth: 3,
    pointRadius: 12,
    pointHoverRadius: 12,
    shadowColor: 'rgba(255, 255, 255, 0.5)',
    shadowBlur: 15
  });

  scatterChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: clusterDatasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Annual Income',
            color: '#94a3b8',
            font: { size: 12, weight: 'bold' }
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.04)'
          },
          ticks: {
            color: '#94a3b8',
            callback: function(value) {
              return formatRupeeCompact(value);
            }
          }
        },
        y: {
          title: {
            display: true,
            text: 'Spending Score (1-100)',
            color: '#94a3b8',
            font: { size: 12, weight: 'bold' }
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.04)'
          },
          ticks: { color: '#94a3b8' }
        }
      },
      plugins: {
        legend: {
          display: false // Using custom legends in HTML
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleColor: '#ffffff',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: function(context) {
              const datasetLabel = context.dataset.label;
              if (datasetLabel === 'Active Profile') {
                return `LIVE USER PROFILE: Income: ${formatRupeeFull(context.raw.x)}, Spending Score: ${context.raw.y}`;
              }
              if (datasetLabel === 'Centroids') {
                return `Cluster Centroid (Income: ${formatRupeeFull(context.raw.x)}, Spending: ${context.raw.y.toFixed(1)})`;
              }
              const raw = context.raw;
              return [
                `Customer ID: ${raw.id}`,
                `Gender: ${raw.gender}, Age: ${raw.age}`,
                `Annual Income: ${formatRupeeFull(raw.x)}`,
                `Spending Score: ${raw.y}`
              ];
            }
          }
        }
      }
    }
  });
}

window.toggleCentroids = function() {
  showCentroids = !showCentroids;
  if (scatterChart) {
    const centroidDataset = scatterChart.data.datasets.find(d => d.label === 'Centroids');
    if (centroidDataset) {
      centroidDataset.hidden = !showCentroids;
      scatterChart.update();
    }
  }
};

window.resetChartZoom = function() {
  if (scatterChart) {
    scatterChart.reset();
  }
};

// --- Database Table Rendering & Utilities ---
function renderDatabaseTable() {
  const tbody = document.getElementById("db-table-body");
  tbody.innerHTML = "";

  const start = (currentPage - 1) * rowsPerPage;
  const end = Math.min(start + rowsPerPage, filteredCustomers.length);
  const pageItems = filteredCustomers.slice(start, end);

  if (pageItems.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">No matching customers found.</td></tr>`;
    document.getElementById("pagination-summary").textContent = "Showing 0 entries";
    renderPaginationControls();
    return;
  }

  pageItems.forEach(c => {
    const tr = document.createElement("tr");
    const segment = SEGMENTS[c.cluster];
    
    tr.innerHTML = `
      <td>#${c.id}</td>
      <td><i data-lucide="${c.gender.toLowerCase() === 'female' ? 'female' : 'male'}" style="width: 14px; height: 14px; vertical-align: middle; margin-right: 4px; color: ${c.gender.toLowerCase() === 'female' ? '#f472b6' : '#60a5fa'};"></i> ${c.gender}</td>
      <td>${c.age}</td>
      <td>${formatRupeeFull(c.income)}</td>
      <td>${c.spending}</td>
      <td><span class="badge" style="background: ${segment.bgColor}; color: ${segment.color}; border: 1px solid rgba(255,255,255,0.03);">${segment.name}</span></td>
    `;
    tbody.appendChild(tr);
  });

  lucide.createIcons();
  
  // Update footer info text
  document.getElementById("pagination-summary").textContent = 
    `Showing ${start + 1} to ${end} of ${filteredCustomers.length} entries`;

  renderPaginationControls();
}

function renderPaginationControls() {
  const totalPages = Math.ceil(filteredCustomers.length / rowsPerPage);
  const container = document.getElementById("page-numbers-container");
  container.innerHTML = "";

  document.getElementById("prev-page-btn").disabled = (currentPage === 1);
  document.getElementById("next-page-btn").disabled = (currentPage === totalPages || totalPages === 0);

  if (totalPages === 0) return;

  // Render max 5 page buttons
  let startPage = Math.max(1, currentPage - 2);
  let endPage = Math.min(totalPages, startPage + 4);
  if (endPage - startPage < 4) {
    startPage = Math.max(1, endPage - 4);
  }

  for (let i = startPage; i <= endPage; i++) {
    const btn = document.createElement("button");
    btn.className = `page-number-btn ${i === currentPage ? 'active' : ''}`;
    btn.textContent = i;
    btn.onclick = () => {
      currentPage = i;
      renderDatabaseTable();
    };
    container.appendChild(btn);
  }
}

window.prevPage = function() {
  if (currentPage > 1) {
    currentPage--;
    renderDatabaseTable();
  }
};

window.nextPage = function() {
  const totalPages = Math.ceil(filteredCustomers.length / rowsPerPage);
  if (currentPage < totalPages) {
    currentPage++;
    renderDatabaseTable();
  }
};

window.handleSearch = function() {
  const query = document.getElementById("db-search").value.toLowerCase().trim();
  filterAndSearchData(query, document.getElementById("db-filter-cluster").value);
};

window.handleFilter = function() {
  const clusterFilter = document.getElementById("db-filter-cluster").value;
  filterAndSearchData(document.getElementById("db-search").value.toLowerCase().trim(), clusterFilter);
};

function filterAndSearchData(searchQuery, clusterFilter) {
  filteredCustomers = dbCustomers.filter(c => {
    const matchesSearch = c.id.toString().includes(searchQuery) || c.gender.toLowerCase().includes(searchQuery);
    const matchesCluster = (clusterFilter === "all" || c.cluster.toString() === clusterFilter);
    return matchesSearch && matchesCluster;
  });

  currentPage = 1;
  renderDatabaseTable();
}

// Table column sorting
window.sortTable = function(colIndex) {
  if (sortColumn === colIndex) {
    sortDirection = -sortDirection;
  } else {
    sortColumn = colIndex;
    sortDirection = 1;
  }

  // Update header arrows styling visual cues (optional)
  filteredCustomers.sort((a, b) => {
    let valA, valB;
    switch(colIndex) {
      case 0: valA = a.id; valB = b.id; break;
      case 1: valA = a.gender; valB = b.gender; break;
      case 2: valA = a.age; valB = b.age; break;
      case 3: valA = a.income; valB = b.income; break;
      case 4: valA = a.spending; valB = b.spending; break;
      case 5: valA = a.cluster; valB = b.cluster; break;
    }

    if (valA < valB) return -sortDirection;
    if (valA > valB) return sortDirection;
    return 0;
  });

  currentPage = 1;
  renderDatabaseTable();
};

// --- Batch Upload (Drag-and-Drop) Setup ---
function setupDragAndDrop() {
  const dropZone = document.getElementById("drag-drop-zone");
  const fileInput = document.getElementById("csv-file-input");

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, () => dropZone.classList.remove("dragover"));
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      processUploadedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      processUploadedFile(fileInput.files[0]);
    }
  });
}

function processUploadedFile(file) {
  if (!file.name.endsWith(".csv")) {
    alert("Please upload a valid CSV file (.csv format only)");
    return;
  }

  const reader = new FileReader();
  reader.onload = function(e) {
    const text = e.target.result;
    const parsed = parseCSV(text);
    
    // Validate schema
    const incomeKey = getIncomeColumnName(parsed[0]);
    if (parsed.length === 0 || !incomeKey || !("Spending Score (1-100)" in parsed[0])) {
      alert("Error: CSV must contain an 'Annual Income' column and 'Spending Score (1-100)' column.");
      return;
    }

    // Process and predict
    uploadedCSVData = parsed.map((c, index) => {
      const id = c["CustomerID"] || (index + 1);
      const gender = c["Gender"] || "Female";
      const age = c["Age"] ? parseInt(c["Age"]) : 30;
      const rawIncome = c[incomeKey];
      const income = normalizeIncomeToKUsd(rawIncome, incomeKey);
      const spending = parseInt(c["Spending Score (1-100)"]);
      const cluster = predictCluster(income, spending);
      return { id, gender, age, income, spending, cluster };
    });

    // Populate Batch Dashboard
    document.getElementById("batch-filename").textContent = `${file.name} - ${uploadedCSVData.length} records processed`;
    document.getElementById("batch-total-count").textContent = uploadedCSVData.length;

    // Calculate details
    const counts = [0, 0, 0, 0, 0];
    let totalSpending = 0;
    uploadedCSVData.forEach(c => {
      counts[c.cluster]++;
      totalSpending += c.spending;
    });

    const dominantClusterIndex = counts.indexOf(Math.max(...counts));
    document.getElementById("batch-dominant-segment").textContent = SEGMENTS[dominantClusterIndex].name;
    document.getElementById("batch-dominant-segment").style.color = SEGMENTS[dominantClusterIndex].color;
    document.getElementById("batch-avg-spending").textContent = (totalSpending / uploadedCSVData.length).toFixed(1);

    // Populate Preview Table (first 10 records)
    const previewBody = document.querySelector("#batch-preview-table tbody");
    previewBody.innerHTML = "";
    uploadedCSVData.slice(0, 10).forEach(c => {
      const tr = document.createElement("tr");
      const segment = SEGMENTS[c.cluster];
      tr.innerHTML = `
        <td>#${c.id}</td>
        <td>${c.gender}</td>
        <td>${c.age}</td>
        <td>${formatRupeeFull(c.income)}</td>
        <td>${c.spending}</td>
        <td><span class="badge" style="background: ${segment.bgColor}; color: ${segment.color};">${segment.name}</span></td>
      `;
      previewBody.appendChild(tr);
    });

    // Draw batch distribution donut chart
    drawBatchChart(counts);

    // Show result view
    document.getElementById("batch-results-view").style.display = "block";
  };

  reader.readAsText(file);
}

function drawBatchChart(counts) {
  const ctx = document.getElementById("batchDistributionChart").getContext("2d");

  if (batchChart) {
    batchChart.destroy();
  }

  batchChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.values(SEGMENTS).map(s => s.name),
      datasets: [{
        data: counts,
        backgroundColor: Object.values(SEGMENTS).map(s => s.color),
        borderColor: '#0b0f19',
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: {
            color: '#94a3b8',
            font: { size: 9 },
            boxWidth: 10
          }
        }
      }
    }
  });
}

// Download template helper
window.downloadTemplate = function() {
  const content = "CustomerID,Gender,Age,Annual Income (Lakhs),Spending Score (1-100)\n1,Female,23,13.28,77\n2,Male,45,66.40,20\n3,Female,32,45.65,50\n";
  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "AuraSeg_Template.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

// Export labeled CSV
window.downloadProcessedCSV = function() {
  if (uploadedCSVData.length === 0) return;

  let content = "CustomerID,Gender,Age,Annual Income (Lakhs),Spending Score (1-100),SegmentID,SegmentName\n";
  uploadedCSVData.forEach(c => {
    const incomeLakhs = (c.income * 0.83).toFixed(2);
    content += `${c.id},${c.gender},${c.age},${incomeLakhs},${c.spending},${c.cluster},"${SEGMENTS[c.cluster].name}"\n`;
  });

  const blob = new Blob([content], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "segmented_customers.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
