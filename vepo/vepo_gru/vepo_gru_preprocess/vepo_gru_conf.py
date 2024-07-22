# 滑动窗进行轨迹划分
len_train = 10
len_label = 5
batch_size = 128

# 地球半径
EARTH_RADIUS = 6378.137

# 超参数
# SOG速度阈值(单位Km/h)
SOG_threshold = 100
# 定义等时间差阈值为10min, 由于是13位时间戳，定义间隔60s需要*1000*10
TIMESTAMP_interval = 60 * 1000 * 10
# 定义最后一个等间隔时间戳与真实时间戳的差值阈值
TIMESTAMP_difference = 100 * 1000 * 10
# 相邻时间间隔超过阈值1h时，以此切分轨迹段
divided_TIME = 1
# 判定行驶到停留点的阈值，插值后的相邻两点距离小于100m，判断为行驶到停留点，当相邻两点超过100m，为航行状态，单位Km
stop_DISTANCE = 100 * 0.001
# 连续三个停留点在停驻段才算停留，小于三合并到航行段
parking_number = 3
