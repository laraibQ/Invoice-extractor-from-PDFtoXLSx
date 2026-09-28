document.addEventListener("DOMContentLoaded", function () {
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const fileName = document.getElementById("file-name");
    const extractBtn = document.getElementById("extract-btn");
    const uploadForm = document.getElementById("upload-form");
    const clientFlash = document.getElementById("client-flash");
    const defaultBtnLabel = "Extract table";

    if (!uploadArea || !fileInput || !fileName || !extractBtn || !uploadForm) {
        return;
    }

    ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    ["dragenter", "dragover"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add("highlight"), false);
    });

    ["dragleave", "drop"].forEach((eventName) => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove("highlight"), false);
    });

    uploadArea.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFiles(files);
        }
    });

    fileInput.addEventListener("change", () => handleFiles(fileInput.files));

    uploadArea.addEventListener("click", (e) => {
        if (e.target.closest("label")) {
            return;
        }
        fileInput.click();
    });

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        if (!fileInput.files || fileInput.files.length === 0) {
            return;
        }

        setProcessing(true);
        clearClientFlash();

        try {
            const response = await fetch(uploadForm.action, {
                method: "POST",
                body: new FormData(uploadForm),
            });

            const contentType = (response.headers.get("content-type") || "").toLowerCase();

            if (!response.ok) {
                throw new Error("Extraction failed. Please try again.");
            }

            // Error redirects return HTML instead of a file.
            if (contentType.includes("text/html")) {
                const html = await response.text();
                const message = extractFlashMessage(html) || "No valid table found in the invoice.";
                showClientFlash(message, "danger");
                return;
            }

            const blob = await response.blob();
            const downloadName = getDownloadName(response) || fallbackDownloadName();
            triggerDownload(blob, downloadName);
        } catch (err) {
            showClientFlash(err.message || "Something went wrong.", "danger");
        } finally {
            setProcessing(false);
        }
    });

    function setProcessing(isProcessing) {
        extractBtn.disabled = isProcessing || !fileInput.files.length;
        extractBtn.innerHTML = isProcessing
            ? '<span class="spinner" aria-hidden="true"></span>Processing…'
            : defaultBtnLabel;
    }

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function handleFiles(files) {
        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];
        if (file.type !== "application/pdf") {
            fileName.textContent = "Please upload a PDF file.";
            fileName.classList.add("is-error");
            extractBtn.disabled = true;
            return;
        }

        fileName.textContent = file.name;
        fileName.classList.remove("is-error");
        extractBtn.disabled = false;
        clearClientFlash();
    }

    function getDownloadName(response) {
        const disposition = response.headers.get("content-disposition") || "";
        const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i);
        if (utfMatch) {
            return decodeURIComponent(utfMatch[1]);
        }
        const plainMatch = disposition.match(/filename="?([^";]+)"?/i);
        return plainMatch ? plainMatch[1] : null;
    }

    function fallbackDownloadName() {
        const format = (uploadForm.querySelector('input[name="format"]:checked') || {}).value || "xlsx";
        const base = (fileInput.files[0].name || "invoice").replace(/\.pdf$/i, "");
        return `${base}.${format}`;
    }

    function triggerDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    function showClientFlash(message, category) {
        if (!clientFlash) {
            return;
        }
        clientFlash.hidden = false;
        clientFlash.innerHTML = `<div class="flash flash-${category}">${message}</div>`;
    }

    function clearClientFlash() {
        if (!clientFlash) {
            return;
        }
        clientFlash.hidden = true;
        clientFlash.innerHTML = "";
    }

    function extractFlashMessage(html) {
        const match = html.match(/class="flash[^"]*"[^>]*>([^<]+)</i);
        return match ? match[1].trim() : null;
    }

    document.querySelectorAll(".feature-item").forEach((feature, index) => {
        setTimeout(() => feature.classList.add("feature-animate"), index * 90);
    });
});
