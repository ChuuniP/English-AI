// Lắng nghe sự kiện chọn chữ (mouseup, keyup, touchend) trên toàn document
// Sử dụng cơ chế Event Delegation để tránh việc phần tử đọc sách bị re-render làm mất sự kiện.
document.addEventListener("mouseup", handleSelectionChange);
document.addEventListener("keyup", handleSelectionChange);
document.addEventListener("touchend", handleSelectionChange);

function handleSelectionChange() {
    // Tìm khung đọc sách đang hoạt động
    const readingZone = document.getElementById("reading-zone-active");
    if (!readingZone) {
        return;
    }

    const selection = window.getSelection();
    if (!selection) {
        return;
    }

    const selectedText = selection.toString().trim();

    // Chỉ xử lý nếu có từ được chọn và vùng bôi đen nằm bên trong khung đọc sách
    if (selectedText && selection.anchorNode && readingZone.contains(selection.anchorNode)) {
        console.log("📌 Từ được chọn:", selectedText);

        // 1. Cập nhật trực tiếp vào ô selected_word_txt để hiển thị tức thì
        const selectedWordContainer = document.getElementById("selected_word_txt");
        if (selectedWordContainer) {
            const inputElement = selectedWordContainer.querySelector("textarea") || selectedWordContainer.querySelector("input");
            if (inputElement) {
                inputElement.value = selectedText;
                // Kích hoạt sự kiện input để Gradio nhận biết thay đổi
                inputElement.dispatchEvent(new Event("input", { bubbles: true }));
            }
        }

        // 2. Đồng thời cập nhật vào hidden_trigger_vocab đề phòng các xử lý phía backend (nếu có)
        const hiddenContainer = document.getElementById("hidden_trigger_vocab");
        if (hiddenContainer) {
            const hiddenInput = hiddenContainer.querySelector("textarea") || hiddenContainer.querySelector("input");
            if (hiddenInput) {
                hiddenInput.value = selectedText;
                hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
            }
        }
    }
}


// Hàm tìm phần tử theo id, xuyên qua được Shadow DOM (Gradio thường bọc UI
// trong custom element <gradio-app> có shadowRoot, nên document.getElementById
// thông thường có thể không thấy được các phần tử bên trong).
function findElementDeep(id, root = document) {
    let el = root.getElementById ? root.getElementById(id) : null;
    if (el) return el;

    const walker = (root.querySelectorAll ? root : document).querySelectorAll("*");
    for (const node of walker) {
        if (node.shadowRoot) {
            const found = findElementDeep(id, node.shadowRoot);
            if (found) return found;
        }
    }
    return null;
}

function findFileInputDeep(container) {
    if (!container) return null;
    // Thử tìm trực tiếp trong light DOM của container trước
    let input = container.querySelector('input[type="file"]');
    if (input) return input;

    // Nếu container (hoặc con của nó) lại có shadow root riêng, quét tiếp vào trong
    const all = container.querySelectorAll("*");
    for (const node of all) {
        if (node.shadowRoot) {
            input = node.shadowRoot.querySelector('input[type="file"]');
            if (input) return input;
        }
    }
    return null;
}

let recorder = null;
let chunks = [];

function initRecorder(){
    const btn = document.getElementById("record-btn");
    if(!btn) return;

    if(btn.dataset.loaded === "1") return;
    btn.dataset.loaded = "1";

    btn.onclick = async () => {
        // =========================
        // START RECORD
        // =========================
        if(recorder == null){
            try {
                console.log("🎙️ [1] Đang xin quyền micro...");
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log("✅ [2] Đã có quyền micro, bắt đầu ghi âm.");
                recorder = new MediaRecorder(stream);
                chunks = [];

                recorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunks.push(e.data);
                    }
                };

                recorder.onstop = () => {
                    console.log(`🛑 [3] Đã dừng ghi âm, số chunk: ${chunks.length}`);
                    const blob = new Blob(chunks, { type: "audio/webm" });
                    console.log(`📦 [4] Tạo blob, kích thước: ${blob.size} bytes`);

                    // Tên file có timestamp để tránh trường hợp trình duyệt/Gradio
                    // coi 2 lần ghi âm liên tiếp là "cùng 1 file" và không bắn change.
                    const file = new File([blob], `record_${Date.now()}.webm`, { type: "audio/webm" });

                    // Tìm input file trong component gr.File ẩn ("hidden-audio-recorder"),
                    // xuyên qua Shadow DOM nếu cần.
                    const container = findElementDeep("hidden-audio-recorder");
                    console.log("🔍 [5] Container #hidden-audio-recorder:", container);

                    const input = findFileInputDeep(container);
                    console.log("🔍 [6] Input file tìm được:", input);

                    if (input) {
                        // Reset trước để đảm bảo sự kiện change luôn được kích hoạt lại
                        input.value = "";

                        const dt = new DataTransfer();
                        dt.items.add(file);
                        input.files = dt.files;

                        // Dispatch cả input lẫn change để chắc chắn Gradio (Svelte) bắt được
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                        input.dispatchEvent(new Event("change", { bubbles: true }));
                        console.log("🚀 [7] Đã dispatch input/change lên Gradio.");
                    } else {
                        console.error("❌ Không tìm thấy input[type=file] trong #hidden-audio-recorder");
                        if (container) {
                            console.log("📋 Cấu trúc bên trong container:", container.outerHTML.slice(0, 2000));
                        }
                    }

                    recorder = null;
                    btn.innerHTML = "🎤 Record";
                    btn.classList.remove("recording");
                };

                recorder.start();
                btn.innerHTML = "■ Stop";
                btn.classList.add("recording");
            } catch (err) {
                console.error("❌ Không thể truy cập Microphone:", err);
            }
        }
        // =========================
        // STOP RECORD
        // =========================
        else {
            console.log("⏹️ Bấm nút Stop...");
            recorder.stop();
            // Dừng tất cả các track micro để tắt đèn báo ghi âm trên trình duyệt
            recorder.stream.getTracks().forEach(track => track.stop());
        }
    };
}

setInterval(initRecorder, 500);