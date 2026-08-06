// Lắng nghe sự kiện chọn chữ (mouseup, keyup, touchend) trên toàn document
// Sử dụng cơ chế Event Delegation để tránh việc phần tử đọc sách bị re-render làm mất sự kiện.
document.addEventListener("mouseup", handleSelectionChange);
document.addEventListener("keyup", handleSelectionChange);
document.addEventListener("touchend", handleSelectionChange);

function handleSelectionChange() {
    // Tìm khung đọc sách đang hoạt động
    // FIX: dùng findElementDeep thay vì document.getElementById thẳng, vì
    // Gradio thường bọc UI trong Shadow DOM (xem comment của findElementDeep
    // bên dưới) -> getElementById thường có thể không thấy được phần tử này,
    // khiến cả tính năng bôi đen tra nghĩa im lặng không chạy.
    const readingZone = findElementDeep("reading-zone-active");
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
        const selectedWordContainer = findElementDeep("selected_word_txt");
        if (selectedWordContainer) {
            const inputElement = selectedWordContainer.querySelector("textarea") || selectedWordContainer.querySelector("input");
            if (inputElement) {
                inputElement.value = selectedText;
                // Kích hoạt sự kiện input để Gradio nhận biết thay đổi
                inputElement.dispatchEvent(new Event("input", { bubbles: true }));
            }
        }

        // 2. Đồng thời cập nhật vào hidden_trigger_vocab đề phòng các xử lý phía backend (nếu có)
        const hiddenContainer = findElementDeep("hidden_trigger_vocab");
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

function setupRecordButton(btn){
    if(btn.dataset.loaded === "1") return;
    btn.dataset.loaded = "1";

    // Nút Word gốc không có data-target -> mặc định dùng "hidden-audio-recorder"
    // để không phá hành vi cũ. Nút Sentence có data-target="hidden-audio-recorder-sentence".
    const targetId = btn.dataset.target || "hidden-audio-recorder";

    let recorder = null;
    let chunks = [];

    btn.onclick = async () => {
        // =========================
        // START RECORD
        // =========================
        if(recorder == null){
            try {
                console.log(`🎙️ [1] (${targetId}) Đang xin quyền micro...`);
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log(`✅ [2] (${targetId}) Đã có quyền micro, bắt đầu ghi âm.`);
                recorder = new MediaRecorder(stream);
                chunks = [];

                recorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunks.push(e.data);
                    }
                };

                recorder.onstop = () => {
                    btn.disabled = false; // mở lại nút sau khi đã dừng xong hẳn
                    console.log(`🛑 [3] (${targetId}) Đã dừng ghi âm, số chunk: ${chunks.length}`);
                    const blob = new Blob(chunks, { type: "audio/webm" });
                    console.log(`📦 [4] (${targetId}) Tạo blob, kích thước: ${blob.size} bytes`);

                    // Tên file có timestamp để tránh trường hợp trình duyệt/Gradio
                    // coi 2 lần ghi âm liên tiếp là "cùng 1 file" và không bắn change.
                    const file = new File([blob], `record_${Date.now()}.webm`, { type: "audio/webm" });

                    // Tìm input file trong đúng component gr.File ẩn ứng với nút này,
                    // xuyên qua Shadow DOM nếu cần.
                    const container = findElementDeep(targetId);
                    console.log(`🔍 [5] (${targetId}) Container:`, container);

                    const input = findFileInputDeep(container);
                    console.log(`🔍 [6] (${targetId}) Input file tìm được:`, input);

                    if (input) {
                        // Reset trước để đảm bảo sự kiện change luôn được kích hoạt lại
                        input.value = "";

                        const dt = new DataTransfer();
                        dt.items.add(file);
                        input.files = dt.files;

                        // Dispatch cả input lẫn change để chắc chắn Gradio (Svelte) bắt được
                        input.dispatchEvent(new Event("input", { bubbles: true }));
                        input.dispatchEvent(new Event("change", { bubbles: true }));
                        console.log(`🚀 [7] (${targetId}) Đã dispatch input/change lên Gradio.`);
                    } else {
                        console.error(`❌ Không tìm thấy input[type=file] trong #${targetId}`);
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
                console.error(`❌ (${targetId}) Không thể truy cập Microphone:`, err);
            }
        }
        // =========================
        // STOP RECORD
        // =========================
        else {
            console.log(`⏹️ (${targetId}) Bấm nút Stop...`);
            // FIX: khóa nút ngay để tránh double-click gọi recorder.stop() 2 lần
            // trước khi onstop (bất đồng bộ) kịp chạy xong -> tránh InvalidStateError.
            // Không ảnh hưởng hành vi cũ: nút vẫn mở lại ngay khi onstop xử lý xong.
            btn.disabled = true;
            recorder.stop();
            // Dừng tất cả các track micro để tắt đèn báo ghi âm trên trình duyệt
            recorder.stream.getTracks().forEach(track => track.stop());
        }
    };
}

function initRecorder(){
    // Gắn sự kiện cho TẤT CẢ các nút có class "record-btn" (Word, Sentence, và
    // các tab sau này nếu thêm) — mỗi nút tự tìm ô hidden-audio-recorder riêng
    // qua thuộc tính data-target.
    document.querySelectorAll(".record-btn").forEach(setupRecordButton);
}

setInterval(initRecorder, 500);