from heatmappy import Heatmapper
from PIL import Image

def heatmap(team):
    example_img_path = 'C:/Users/ch061/PycharmProjects/pythonProject5/venv/groundteam1.jpg'
    example_img = Image.open(example_img_path)

    # 이미지 사이즈 튜플 형태로 반환 (w, h)
    image_size = example_img.size
    w = image_size[0]
    h = image_size[1]

    coordinate = []
    for i in team:
        coordinate.append(((i[0] + i[2] * 1.7), (i[1] + i[3]) * 1.6))
    example_points = coordinate  # 히트맵 중심 좌표 설정
    # 히트맵 그리기
    heatmapper = Heatmapper(
        point_diameter=50,  # the size of each point to be drawn
        point_strength=1,  # the strength, between 0 and 1, of each point to be drawn
        opacity=0.4,  # the opacity of the heatmap layer
        colours='default',  # 'default' or 'reveal'
        # OR a matplotlib LinearSegmentedColorMap object
        # OR the path to a horizontal scale image
        grey_heatmapper='PIL'  # The object responsible for drawing the points
        # Pillow used by default, 'PySide' option available if installed
    )

    # 이미지 위에 히트맵 그리기
    heatmap = heatmapper.heatmap_on_img(example_points, example_img)
    # 출력 이미지 경로 설정
    heatmap.save('C:/Users/ch061/PycharmProjects/pythonProject5/venv/result.png')