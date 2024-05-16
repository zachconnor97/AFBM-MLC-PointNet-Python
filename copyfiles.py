import shutil
import pandas as pd

filename = "C:/Users/gabri/Downloads/output-obj_cloud_out10k1_test.csv"

df = pd.read_csv(filename)

filetocopy = "D:/ZachResearch\MeshlabPython and Files/Objs/ashcan_trash can_garbage can_wastebin_ash bin_ash-bin_ashbin_dustbin_trash barrel_trash bin_2747177_50978d3850fcf4e371a126804ae24042_.obj"

desination = "D:/ZachResearch/MeshlabPython and Files/Copies/ashcan_trash can_garbage can_wastebin_ash bin_ash-bin_ashbin_dustbin_trash barrel_trash bin_copy_2747177_50978d3850fcf4e371a126804ae24042_.obj"

shutil.copy(filetocopy, desination)