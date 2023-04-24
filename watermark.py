from PIL import Image, ImageDraw, ImageFont
import cv2
import PyPDF2

def add_watermark(image_path, watermark_text):
    # Mở ảnh và tạo đối tượng Image từ đường dẫn ảnh
    image = Image.open(image_path)

    # Tạo một đối tượng ImageDraw để vẽ watermark lên ảnh
    draw = ImageDraw.Draw(image)

    # Tạo đối tượng Font và đặt kích thước của nó
    font = ImageFont.truetype('impact.ttf', 36)

    # Tính toán kích thước của watermark
    left, top, right, bottom = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = right - left
    text_height = bottom - top

    # Tính toán vị trí để đặt watermark ở giữa ảnh
    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2

    # Đặt màu cho watermark (ở đây là màu đen)
    text_color = (255, 255, 255)

    # Vẽ watermark lên ảnh
    draw.text((x, y), watermark_text, fill=text_color, font=font)

    # Lưu ảnh mới đã có watermark
    image.save('watermarked_image.jpg')

def add_watermark2():
    # Tải video và mở video để đọc
    video = cv2.VideoCapture('video_input.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Tạo video writer để ghi video đầu ra
    video_out = cv2.VideoWriter('video_output.mp4', fourcc, 30, (1920, 1080))

    # Tải hình ảnh watermark và chuyển đổi sang định dạng RGBA
    watermark = cv2.imread('watermark.png', cv2.IMREAD_UNCHANGED)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGBA)

    # Lấy kích thước của watermark và điều chỉnh kích thước nếu cần thiết
    (wm_height, wm_width, wm_channels) = watermark.shape
    if wm_height > 100 or wm_width > 100:
        watermark = cv2.resize(watermark, (0, 0), fx=0.5, fy=0.5)

    # Lặp lại từng khung hình trong video và thêm watermark vào đó
    while True:
        # Đọc từng khung hình
        ret, frame = video.read()
        if not ret:
            break

        # Thêm watermark vào khung hình
        overlay = cv2.resize(watermark, (0, 0), fx=0.5, fy=0.5)
        (wm_height, wm_width, wm_channels) = overlay.shape
        (frame_height, frame_width, frame_channels) = frame.shape
        x_pos = frame_width - wm_width - 10
        y_pos = frame_height - wm_height - 10
        alpha_s = overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            frame[y_pos:y_pos + wm_height, x_pos:x_pos + wm_width, c] = (alpha_s * overlay[:, :, c] +
                                                                         alpha_l * frame[y_pos:y_pos + wm_height,
                                                                                   x_pos:x_pos + wm_width, c])

        # Ghi khung hình mới có chứa watermark vào video đầu ra
        video_out.write(frame)

    # Giải phóng các tài nguyên
    video.release()
    video_out.release()
    cv2.destroyAllWindows()

def add_watermark3():
    # Tạo watermark
    watermark = PyPDF2.PdfReader(open('watermark.pdf', 'rb'))

    # Tạo file output
    output = PyPDF2.PdfWriter()

    # Đọc file input
    input_file = PyPDF2.PdfReader(open('input.pdf', 'rb'))

    # Với mỗi trang trong file input, đặt watermark lên trang đó
    for i in range(len(input_file.pages)):
        page = input_file.pages[i]
        page.merge_page(watermark.pages[0])
        output.add_page(page)

    # Lưu kết quả vào file mới
    with open('output.pdf', 'wb') as f:
        output.write(f)

if __name__ == '__main__':
    # add_watermark('image.jpg', 'WATERMARK')
    # add_watermark2()
    add_watermark3()