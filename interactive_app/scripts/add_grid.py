from PIL import Image, ImageDraw, ImageFont

def add_grid_with_padding(image_pil, grid_interval=50, font_size=12, tick_length=5, 
                          padding_left=50, padding_bottom=40, padding_top=10, padding_right=10):
    """
    在图像上添加网格和刻度，并在图像周围增加边距以容纳刻度和标签。

    参数:
    image_pil (PIL.Image.Image): 原始Pillow图像对象。
    grid_interval (int): 网格线的间隔（像素）。
    font_size (int): 刻度标签的字体大小。
    tick_length (int): 刻度线的长度（像素）。
    padding_left (int): 左边距。
    padding_bottom (int): 下边距。
    padding_top (int): 上边距。
    padding_right (int): 右边距。
    """
    original_width, original_height = image_pil.size

    # 1. 边距已作为参数传入，也可以在此处动态计算
    # 例如，根据字体和最大标签文本的尺寸计算

    # 2. 创建新图像画布
    new_width = original_width + padding_left + padding_right
    new_height = original_height + padding_top + padding_bottom
    
    # 确保模式兼容，例如转换为RGB
    # 如果原始图像有透明度 (RGBA)，新画布也应支持透明度以保留
    if image_pil.mode == 'RGBA':
        new_image = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0)) # 透明背景
    elif image_pil.mode == 'P' and 'transparency' in image_pil.info:
        # 如果是P模式且有透明度，转换为RGBA
        image_pil = image_pil.convert('RGBA')
        new_image = Image.new('RGBA', (new_width, new_height), (255, 255, 255, 0))
    else:
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        new_image = Image.new('RGB', (new_width, new_height), 'white') # 白色背景
    
    draw = ImageDraw.Draw(new_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default() # Fallback to default font

    # 3. 将原始图像绘制到新画布上
    image_paste_x = padding_left
    image_paste_y = padding_top
    new_image.paste(image_pil, (image_paste_x, image_paste_y))

    # 4. 调整栅格和刻度的绘制逻辑
    # 原始图像在新画布上的区域边界
    img_area_x0 = padding_left
    img_area_y0 = padding_top
    img_area_x1 = padding_left + original_width
    img_area_y1 = padding_top + original_height

    # --- 绘制栅格线和刻度 ---
    grid_color = (200, 200, 200)  # Light grey for grid
    tick_color = (50, 50, 50)    # Dark grey for ticks and labels

    # 水平栅格线和Y轴刻度
    for y_orig in range(0, original_height + 1, grid_interval):
        # y_orig 是原始图像中的y坐标值
        y_on_new_canvas = img_area_y0 + y_orig # 在新画布上对应的y位置

        # 绘制水平栅格线 (横跨原图区域)
        if y_orig <= original_height : #确保线在图像区域内
             draw.line([(img_area_x0, y_on_new_canvas), (img_area_x1, y_on_new_canvas)], fill=grid_color)
        
        # Y轴刻度和标签
        if y_orig <= original_height: # 刻度值对应原图
            label_text = str(y_orig)
            try: # Pillow >= 9.2.0 for textbbox
                bbox = draw.textbbox((0,0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError: # Older Pillow versions use textsize
                 text_width, text_height = draw.textsize(label_text, font=font)

            # 刻度线 (在左边距)
            tick_start_x = padding_left - tick_length
            tick_end_x = padding_left
            draw.line([(tick_start_x, y_on_new_canvas), (tick_end_x, y_on_new_canvas)], fill=tick_color)
            
            # 标签 (在刻度线左侧)
            label_x = padding_left - tick_length - text_width - 5 # 5px 间距
            label_y = y_on_new_canvas - (text_height / 2)
            draw.text((label_x, label_y), label_text, fill=tick_color, font=font)

    # 垂直栅格线和X轴刻度
    for x_orig in range(0, original_width + 1, grid_interval):
        # x_orig 是原始图像中的x坐标值
        x_on_new_canvas = img_area_x0 + x_orig # 在新画布上对应的x位置

        # 绘制垂直栅格线 (纵贯原图区域)
        if x_orig <= original_width:
            draw.line([(x_on_new_canvas, img_area_y0), (x_on_new_canvas, img_area_y1)], fill=grid_color)

        # X轴刻度和标签
        if x_orig <= original_width: # 刻度值对应原图
            label_text = str(x_orig)
            try:
                bbox = draw.textbbox((0,0), label_text, font=font)
                text_width = bbox[2] - bbox[0]
                # text_height = bbox[3] - bbox[1] # Not strictly needed for x-axis label y-pos
            except AttributeError:
                 text_width, _ = draw.textsize(label_text, font=font)
            
            # 刻度线 (在下边距)
            tick_start_y = img_area_y1 
            tick_end_y = img_area_y1 + tick_length
            draw.line([(x_on_new_canvas, tick_start_y), (x_on_new_canvas, tick_end_y)], fill=tick_color)
            
            # 标签 (在刻度线下方)
            label_x = x_on_new_canvas - (text_width / 2)
            label_y = img_area_y1 + tick_length + 5 # 5px 间距
            draw.text((label_x, label_y), label_text, fill=tick_color, font=font)
            
    # (可选) 绘制原始图像区域的边框
    draw.rectangle([img_area_x0, img_area_y0, img_area_x1, img_area_y1], outline=tick_color)

    return new_image

try:
    # 假设您有一个名为 "original.png" 的图像文件
    input_image = Image.open(r"C:\Users\17876\Desktop\CK\RoomSketcher-Office-Floor-Plan-PID3529710-2D-bw-with-Labels.jpg") 
    
    # 调用函数，可以调整边距等参数
    output_image = add_grid_with_padding(
        input_image, 
        grid_interval=100, 
        font_size=10, 
        tick_length=5,
        padding_left=60, # 为Y轴标签留出更多空间
        padding_bottom=50, # 为X轴标签留出更多空间
        padding_top=20,
        padding_right=20
    )
    
    output_image.save(r"C:\Users\17876\Downloads\_local_3.png")
    # output_image.show() 
    print("图像已处理并保存")
except FileNotFoundError:
    print("错误: 原始图像文件未找到。")
except Exception as e:
    print(f"处理图像时发生错误: {e}")
