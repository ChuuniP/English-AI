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