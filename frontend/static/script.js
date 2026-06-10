const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");

const predictBtn = document.getElementById("predictBtn");
const clearBtn = document.getElementById("clearBtn");

const predictedDigit = document.getElementById("predictedDigit");
const confidenceScore = document.getElementById("confidenceScore");
const topPredictions = document.getElementById("topPredictions");


let isDrawing = false;

// Set white background initially
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Drawing style
ctx.strokeStyle = "black";
ctx.lineWidth = 18;
ctx.lineCap = "round";

// Mouse events
canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("mouseleave", stopDrawing);

// Touch events
canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
canvas.addEventListener("touchmove", handleTouchMove, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(getX(e), getY(e));
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(getX(e), getY(e));
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function getX(e) {
    const rect = canvas.getBoundingClientRect();
    return e.clientX - rect.left;
}

function getY(e) {
    const rect = canvas.getBoundingClientRect();
    return e.clientY - rect.top;
}

function handleTouchStart(e) {
    e.preventDefault();
    const touch = e.touches[0];
    startDrawing(touch);
}

function handleTouchMove(e) {
    e.preventDefault();
    const touch = e.touches[0];
    draw(touch);
}

clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    predictedDigit.textContent = "-";
    confidenceScore.textContent = "-";
    topPredictions.innerHTML = "";
    heatmapImage.src = "";
});

predictBtn.addEventListener("click", async () => {
    const imageData = canvas.toDataURL("image/png");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ image: imageData })
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        predictedDigit.textContent = data.predicted_digit;
        confidenceScore.textContent = `${data.confidence}%`;

        topPredictions.innerHTML = "";
        data.top_3_predictions.forEach(item => {
            const li = document.createElement("li");
            li.textContent = `Digit ${item.digit}: ${item.confidence}%`;
            topPredictions.appendChild(li);
        });

       
    } catch (error) {
        console.error("Prediction error:", error);
        alert("Something went wrong while predicting.");
    }
});

// Analytics dashboard
async function loadAnalytics() {
    try {
        const response = await fetch("/analytics");
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        document.getElementById("modelAccuracy").textContent = `${(data.overall_accuracy * 100).toFixed(2)}%`;
        document.getElementById("modelLoss").textContent = data.overall_loss.toFixed(4);

        renderClassAccuracyChart(data.class_accuracy);
        renderTrainingChart(data.training_history);
        renderConfusionMatrix(data.confusion_matrix);
    } catch (error) {
        console.error("Analytics loading error:", error);
    }
}

function renderClassAccuracyChart(classAccuracy) {
    const labels = Object.keys(classAccuracy);
    const values = Object.values(classAccuracy).map(v => v * 100);

    new Chart(document.getElementById("classAccuracyChart"), {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Accuracy (%)",
                data: values,
                backgroundColor: "#2d6cdf"
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

function renderTrainingChart(history) {
    const labels = history.accuracy.map((_, i) => `Epoch ${i + 1}`);

    new Chart(document.getElementById("trainingChart"), {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Train Accuracy",
                    data: history.accuracy,
                    borderColor: "green",
                    fill: false
                },
                {
                    label: "Validation Accuracy",
                    data: history.val_accuracy,
                    borderColor: "orange",
                    fill: false
                }
            ]
        },
        options: {
            responsive: true
        }
    });
}

function renderConfusionMatrix(matrix) {
    const container = document.getElementById("confusionMatrixTable");

    let html = "<table><tr><th>Actual \\ Pred</th>";
    for (let i = 0; i < 10; i++) {
        html += `<th>${i}</th>`;
    }
    html += "</tr>";

    matrix.forEach((row, i) => {
        html += `<tr><th>${i}</th>`;
        row.forEach(cell => {
            html += `<td>${cell}</td>`;
        });
        html += "</tr>";
    });

    html += "</table>";
    container.innerHTML = html;
}

loadAnalytics();
