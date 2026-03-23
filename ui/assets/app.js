const form = document.getElementById("hair-form");
const imageInput = document.getElementById("image-input");
const promptInput = document.getElementById("prompt-input");
const fileMeta = document.getElementById("file-meta");
const promptCounter = document.getElementById("prompt-counter");
const originalPreview = document.getElementById("original-preview");
const resultPreview = document.getElementById("result-preview");
const originalEmpty = document.getElementById("original-empty");
const resultEmpty = document.getElementById("result-empty");
const loadingState = document.getElementById("loading-state");
const submitButton = document.getElementById("submit-button");
const clearButton = document.getElementById("clear-button");
const statusPill = document.getElementById("status-pill");
const responseMessage = document.getElementById("response-message");
const downloadLink = document.getElementById("download-link");
const dropzone = document.getElementById("dropzone");
const promptChips = document.querySelectorAll(".prompt-chip");
const stepUpload = document.getElementById("step-upload");
const stepPrompt = document.getElementById("step-prompt");
const stepGenerate = document.getElementById("step-generate");

let originalObjectUrl = null;

function setStepState(element, state) {
    if (!element) {
        return;
    }

    element.classList.remove("is-active", "is-done");

    if (state === "active") {
        element.classList.add("is-active");
    }

    if (state === "done") {
        element.classList.add("is-done");
    }
}

function syncWorkflow({ loading = false } = {}) {
    const hasImage = Boolean(imageInput.files?.[0]);
    const hasPrompt = promptInput.value.trim().length > 0;
    const hasResult = Boolean(resultPreview.getAttribute("src"));

    setStepState(stepUpload, hasImage ? "done" : "active");

    if (!hasImage) {
        setStepState(stepPrompt, "");
        setStepState(stepGenerate, "");
        return;
    }

    if (!hasPrompt) {
        setStepState(stepPrompt, "active");
        setStepState(stepGenerate, "");
        return;
    }

    setStepState(stepPrompt, "done");

    if (loading) {
        setStepState(stepGenerate, "active");
        return;
    }

    setStepState(stepGenerate, hasResult ? "done" : "active");
}

function updatePromptCounter() {
    if (!promptCounter) {
        return;
    }

    const maxLength = Number(promptInput.maxLength) || 280;
    promptCounter.textContent = `${promptInput.value.length}/${maxLength}`;
}

function clearObjectUrl() {
    if (!originalObjectUrl) {
        return;
    }

    URL.revokeObjectURL(originalObjectUrl);
    originalObjectUrl = null;
}

function formatFileSize(bytes) {
    if (bytes >= 1024 * 1024) {
        return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }

    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function setStatus(text, variant = "") {
    statusPill.textContent = text;
    statusPill.className = "status-pill";
    if (variant) {
        statusPill.classList.add(variant);
    }
}

function updateOriginalPreview(file) {
    clearObjectUrl();

    if (!file) {
        originalPreview.removeAttribute("src");
        originalPreview.classList.add("hidden");
        originalEmpty.classList.remove("hidden");
        fileMeta.textContent = "Ningun archivo seleccionado";
        dropzone.classList.remove("has-file");
        syncWorkflow();
        return;
    }

    if (!file.type.startsWith("image/")) {
        setStatus("Formato no valido", "error");
        responseMessage.textContent = "Sube una imagen valida para continuar.";
        updateOriginalPreview(null);
        return;
    }

    originalObjectUrl = URL.createObjectURL(file);
    originalPreview.src = originalObjectUrl;
    originalPreview.classList.remove("hidden");
    originalEmpty.classList.add("hidden");
    fileMeta.textContent = `${file.name} - ${formatFileSize(file.size)}`;
    dropzone.classList.add("has-file");
    setStatus("Imagen lista", "success");
    syncWorkflow();
}

function resetResultState() {
    resultPreview.removeAttribute("src");
    resultPreview.classList.add("hidden");
    resultEmpty.classList.remove("hidden");
    loadingState.classList.add("hidden");
    downloadLink.classList.add("hidden");
    syncWorkflow();
}

function clearForm() {
    form.reset();
    promptInput.value = "";
    responseMessage.textContent = "";
    setStatus("Esperando imagen");
    updatePromptCounter();
    updateOriginalPreview(null);
    resetResultState();
}

imageInput.addEventListener("change", () => {
    updateOriginalPreview(imageInput.files[0]);
    resetResultState();
    responseMessage.textContent = "";
});

["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        setStatus("Formato no valido", "error");
        responseMessage.textContent = "Solo se aceptan imagenes.";
        return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    imageInput.files = transfer.files;
    updateOriginalPreview(file);
    resetResultState();
    responseMessage.textContent = "";
});

promptChips.forEach((chip) => {
    chip.addEventListener("click", () => {
        promptInput.value = chip.dataset.prompt ?? "";
        promptInput.focus();
        updatePromptCounter();
        syncWorkflow();
    });
});

if (clearButton) {
    clearButton.addEventListener("click", clearForm);
}

promptInput.addEventListener("input", () => {
    updatePromptCounter();
    syncWorkflow();
});

promptInput.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files[0];
    const prompt = promptInput.value.trim();

    if (!file) {
        setStatus("Falta imagen", "error");
        responseMessage.textContent = "Selecciona una fotografía antes de generar el resultado.";
        return;
    }

    if (!prompt) {
        setStatus("Falta prompt", "error");
        responseMessage.textContent = "Escribe el estilo o cambio deseado para continuar.";
        promptInput.focus();
        return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("prompt", prompt);

    submitButton.disabled = true;
    if (clearButton) {
        clearButton.disabled = true;
    }
    loadingState.classList.remove("hidden");
    resultEmpty.classList.add("hidden");
    responseMessage.textContent = "";
    setStatus("Generando resultado", "loading");
    syncWorkflow({ loading: true });

    try {
        const response = await fetch("/hair/change", {
            method: "POST",
            body: formData,
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            const detail = Array.isArray(payload.detail)
                ? payload.detail.map((d) => d.msg ?? String(d)).join(" ")
                : payload.detail ?? "No fue posible generar el resultado.";
            throw new Error(detail);
        }

        if (!payload.image_mime_type || !payload.image_base64) {
            throw new Error("La respuesta no contiene una imagen valida.");
        }

        const resultUrl = `data:${payload.image_mime_type};base64,${payload.image_base64}`;
        resultPreview.src = resultUrl;
        resultPreview.classList.remove("hidden");
        loadingState.classList.add("hidden");
        downloadLink.href = resultUrl;
        downloadLink.classList.remove("hidden");
        setStatus("Resultado listo", "success");
        responseMessage.textContent = `${payload.message ?? "Resultado generado"}.`;
        syncWorkflow();
    } catch (error) {
        loadingState.classList.add("hidden");
        resultEmpty.classList.remove("hidden");
        setStatus("Error", "error");
        responseMessage.textContent = error.message;
        syncWorkflow();
    } finally {
        submitButton.disabled = false;
        if (clearButton) {
            clearButton.disabled = false;
        }
    }
});

updatePromptCounter();
updateOriginalPreview(null);
resetResultState();