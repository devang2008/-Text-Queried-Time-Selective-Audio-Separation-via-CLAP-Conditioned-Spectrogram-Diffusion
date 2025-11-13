// Frontend JavaScript for audio separation app

let currentFiles = [];
let currentFileId = null;
let isProcessing = false;
let uploadedFile = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadFiles();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // File selection and upload
    document.getElementById('fileSelect').addEventListener('change', onFileChange);
    document.getElementById('uploadBtn').addEventListener('click', () => {
        document.getElementById('fileUpload').click();
    });
    document.getElementById('fileUpload').addEventListener('change', handleFileUpload);
    
    // Sliders
    document.getElementById('startTime').addEventListener('input', updateStartTimeLabel);
    document.getElementById('endTime').addEventListener('input', updateEndTimeLabel);
    document.getElementById('components').addEventListener('input', updateComponentsLabel);
    
    // Buttons
    document.getElementById('runBtn').addEventListener('click', runSeparation);
    document.getElementById('detectBtn').addEventListener('click', detectSounds);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeAudioWithAI);
}

// Handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const uploadStatus = document.getElementById('uploadStatus');
    uploadStatus.style.display = 'block';
    uploadStatus.textContent = '⏳ Uploading...';
    uploadStatus.style.background = '#ffc107';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        uploadedFile = await response.json();
        
        // Clear file select
        document.getElementById('fileSelect').value = '';
        
        // Update UI
        currentFileId = uploadedFile.id;
        document.getElementById('classInfo').textContent = `Uploaded: ${uploadedFile.filename}`;
        document.getElementById('mixPlayer').src = `/api/audio/${uploadedFile.id}`;
        
        uploadStatus.textContent = `✅ Uploaded: ${uploadedFile.filename}`;
        uploadStatus.style.background = '#28a745';
        
        // Show analysis button for uploaded files
        document.getElementById('analyzeSection').style.display = 'block';
        document.getElementById('analysisResults').style.display = 'none';
        
        // Load spectrogram
        loadSpectrogram(uploadedFile.id);
        
        // Hide results
        document.getElementById('resultsSection').style.display = 'none';
        
        showToast('File uploaded successfully! Click "Detect Audio Content" to analyze.', 'success');
        
    } catch (error) {
        uploadStatus.textContent = `❌ Error: ${error.message}`;
        uploadStatus.style.background = '#dc3545';
        showToast('Upload failed: ' + error.message, 'error');
    }
}

// Analyze audio content
async function analyzeAudioWithAI() {
    if (!currentFileId || !currentFileId.startsWith('upload_')) {
        showToast('Please upload an audio file first', 'error');
        return;
    }
    
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('analyzeLoading');
    const resultsDiv = document.getElementById('analysisResults');
    const contentDiv = document.getElementById('analysisContent');
    
    analyzeBtn.disabled = true;
    loading.style.display = 'block';
    resultsDiv.style.display = 'none';
    
    try {
        const response = await fetch(`/api/analyze?file_id=${currentFileId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Analysis failed');
        }
        
        const result = await response.json();
        
        // Display results - only detected sounds
        let html = '';
        
        // Show only specific sounds
        if (result.specific_sounds && result.specific_sounds.length > 0) {
            html += '<div class="analysis-tags">';
            result.specific_sounds.forEach(sound => {
                html += `<span class="analysis-tag">${sound}</span>`;
            });
            html += '</div>';
        } else {
            html += '<p style="opacity: 0.7;">No sounds detected</p>';
        }
        
        contentDiv.innerHTML = html;
        resultsDiv.style.display = 'block';
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        
        showToast('Content detection complete!', 'success');
        
    } catch (error) {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        
        showToast('Detection failed: ' + error.message, 'error');
    }
}

// Use a suggested prompt
function usePrompt(prompt) {
    document.getElementById('promptInput').value = prompt;
    showToast(`Prompt set: "${prompt}"`, 'success');
    // Scroll to separation section
    document.querySelector('.panel:nth-child(2)').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Load available files from API
async function loadFiles() {
    try {
        const response = await fetch('/api/files');
        if (!response.ok) throw new Error('Failed to load files');
        
        currentFiles = await response.json();
        
        const select = document.getElementById('fileSelect');
        select.innerHTML = '<option value="">-- Select a file --</option>';
        
        currentFiles.forEach(file => {
            const option = document.createElement('option');
            option.value = file.id;
            option.textContent = `${file.id} (${file.class_label})`;
            select.appendChild(option);
        });
        
    } catch (error) {
        showToast('Error loading files: ' + error.message, 'error');
    }
}

// Handle file selection change
async function onFileChange(event) {
    const fileId = event.target.value;
    if (!fileId) {
        currentFileId = null;
        document.getElementById('mixPlayer').src = '';
        return;
    }
    
    // Clear uploaded file status
    uploadedFile = null;
    document.getElementById('uploadStatus').style.display = 'none';
    document.getElementById('fileUpload').value = '';
    
    // Hide AI analysis section for dataset files (they already have labels)
    document.getElementById('analyzeSection').style.display = 'none';
    document.getElementById('analysisResults').style.display = 'none';
    
    currentFileId = fileId;
    const file = currentFiles.find(f => f.id === fileId);
    
    // Update class info
    document.getElementById('classInfo').textContent = `Class: ${file.class_label}`;
    
    // Update audio player with actual audio file
    document.getElementById('mixPlayer').src = file.audio_url || `/api/audio/${fileId}`;
    
    // Load spectrogram
    loadSpectrogram(fileId);
    
    // Hide results section
    document.getElementById('resultsSection').style.display = 'none';
}

// Load and display spectrogram
async function loadSpectrogram(fileId) {
    const img = document.getElementById('mixSpectrogram');
    const loading = document.getElementById('mixLoading');
    
    img.style.display = 'none';
    loading.style.display = 'block';
    
    try {
        const response = await fetch(`/api/spectrogram?file_id=${fileId}`);
        if (!response.ok) throw new Error('Failed to load spectrogram');
        
        const data = await response.json();
        img.src = data.url;
        img.style.display = 'block';
        loading.style.display = 'none';
        
    } catch (error) {
        loading.textContent = 'Error loading spectrogram';
        showToast('Error loading spectrogram: ' + error.message, 'error');
    }
}

// Update slider labels
function updateStartTimeLabel() {
    const value = document.getElementById('startTime').value;
    document.getElementById('startTimeValue').textContent = value;
}

function updateEndTimeLabel() {
    const value = document.getElementById('endTime').value;
    const label = value >= 5 ? `${value}s (full duration)` : `${value}s`;
    document.getElementById('endTimeValue').textContent = label;
}

function updateComponentsLabel() {
    const value = document.getElementById('components').value;
    document.getElementById('componentsValue').textContent = value;
}

// Run audio separation
async function runSeparation() {
    if (isProcessing) return;
    
    // Validate inputs
    if (!currentFileId) {
        showToast('Please select an audio file', 'error');
        return;
    }
    
    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) {
        showToast('Please enter a text prompt', 'error');
        return;
    }
    
    const mode = document.querySelector('input[name="mode"]:checked').value;
    const method = document.querySelector('input[name="method"]:checked').value;
    const t0 = parseFloat(document.getElementById('startTime').value);
    const t1 = parseFloat(document.getElementById('endTime').value);
    const k_components = parseInt(document.getElementById('components').value);
    
    if (t1 <= t0) {
        showToast('End time must be greater than start time', 'error');
        return;
    }
    
    // Show spinner, disable button
    isProcessing = true;
    document.getElementById('runBtn').disabled = true;
    document.getElementById('spinner').style.display = 'block';
    
    try {
        // Choose endpoint based on method
        const endpoint = method === 'unet' ? '/api/separate_unet' : '/api/separate';
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: currentFileId,
                prompt: prompt,
                mode: mode,
                t0: t0,
                t1: t1 >= 5 ? null : t1,  // null means full duration
                k_components: k_components
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Separation failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
        const methodName = method === 'unet' ? 'UNet (Trained Model)' : 'NMF (Baseline)';
        showToast(`Separation completed with ${methodName}!`, 'success');
        
    } catch (error) {
        showToast('Separation error: ' + error.message, 'error');
    } finally {
        isProcessing = false;
        document.getElementById('runBtn').disabled = false;
        document.getElementById('spinner').style.display = 'none';
    }
}

// Display separation results
function displayResults(result) {
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Update audio players
    document.getElementById('outPlayer').src = result.out_wav;
    document.getElementById('residualPlayer').src = result.residual_wav;
    
    // Update images
    document.getElementById('outSpectrogram').src = result.out_spec_png;
    document.getElementById('maskImage').src = result.mask_png;
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Detect sound classes
async function detectSounds() {
    if (!currentFileId) {
        showToast('Please select an audio file', 'error');
        return;
    }
    
    const btn = document.getElementById('detectBtn');
    btn.disabled = true;
    btn.textContent = 'Detecting...';
    
    try {
        const response = await fetch('/api/classes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                file_id: currentFileId,
                k_components: 10
            })
        });
        
        if (!response.ok) throw new Error('Detection failed');
        
        const results = await response.json();
        displayDetectionResults(results);
        
    } catch (error) {
        showToast('Detection error: ' + error.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Detect Classes';
    }
}

// Display detection results as bar chart
function displayDetectionResults(results) {
    const container = document.getElementById('detectionResults');
    container.innerHTML = '';
    
    results.forEach(result => {
        const bar = document.createElement('div');
        bar.className = 'detection-bar';
        
        const label = document.createElement('div');
        label.className = 'detection-label';
        label.innerHTML = `<span>${result.class}</span><span>${(result.score * 100).toFixed(1)}%</span>`;
        
        const progress = document.createElement('div');
        progress.className = 'detection-progress';
        
        const fill = document.createElement('div');
        fill.className = 'detection-fill';
        fill.style.width = `${result.score * 100}%`;
        
        progress.appendChild(fill);
        bar.appendChild(label);
        bar.appendChild(progress);
        container.appendChild(bar);
    });
}

// Show toast notification
function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = 'toast show ' + type;
    
    setTimeout(() => {
        toast.className = 'toast ' + type;
    }, 3000);
}
