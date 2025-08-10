from ..core.Point import Point
from ..core.Reference import Ref_Point, mainRef_Point
import numpy as np
from ..models.PolynomialRegression import PolynomialRegression
from scipy.spatial.distance import euclidean
from ..utils import Constants
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from ..models.NN import SimpleNN
import torch.nn as nn
import torch.optim as optim
import matplotlib
from scipy.interpolate import LSQUnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class ChooseRef:
    def __init__(self, num_ref, cluster, main_pivot, oth_pivot, type=0):
        self.num_ref = num_ref
        self.dim = len(cluster.clu_point[0].coordinate)
        self.num_data = len(cluster.clu_point)
        self.Pos = list(range(self.num_data))  # 初始化 Pos 为每个点的索引列表
        self.mainRefPoint = main_pivot
        self.RefPoint_Set = []
        self.mainRefDisArr = []
        if type == 0:
            self.RefPoint_Set = self.choose_ref_point(cluster.clu_point)
        elif type == 1 and oth_pivot is not None:
            # print(type)
            self.RefPoint_Set = self.choose_ref_point_input(cluster.clu_point, oth_pivot)
        
        self.mainRefDisArr = self.calculate_dis_arr(cluster.clu_point, self.mainRefPoint)

        r = self.caculate_r(cluster, self.mainRefPoint)
        # print(r)
        # r = max(self.mainRefDisArr)
        # print(self.mainRefDisArr[-1])
        # print(r)

        # 不需要dict_circle
        # dict_circle = self.caculate_circle_bound(self.mainRefDisArr)

        self.main_pointSet = mainRef_Point(point=self.mainRefPoint, r=r, r_low=min(self.mainRefDisArr), ref_points=self.RefPoint_Set)
        # self.main_pointSet.dict_circle = dict_circle
        
        self.main_pointSet.set_main_ref_dis_arr(self.mainRefDisArr)
        ########这是多项式
        mainRefPt_coeffs = self.train_model(self.mainRefDisArr, self.Pos)
        self.main_pointSet.set_coeffs(mainRefPt_coeffs)

        ########这是神经网络
        model = self.train_model(self.mainRefDisArr, self.Pos)
        self.main_pointSet.set_model(model)

    def choose_ref_point(self, cluster_points):
        ref_points = []
        for i in range(self.num_ref):
            max_val = -np.inf
            selected_point = None
            for point in cluster_points:
                if i < len(point.coordinate) and point.coordinate[i] > max_val:
                    max_val = point.coordinate[i]
                    selected_point = point

            if selected_point is not None:
                dis = self.calculate_dis_arr(cluster_points, selected_point)
                coeffs = self.train_model(dis, self.Pos)
                # 创建 Ref_Point 实例
                ref_point = Ref_Point(point=selected_point, r=max(dis), r_low=min(dis))
                # 使用设置方法分配 dis 和 coeffs
                ref_point.set_dis_arr(dis)
                ref_point.set_coeffs(coeffs)
                # ref_point.set_model(coeffs)
                ref_points.append(ref_point)

        return ref_points

    def choose_ref_point_input(self, cluster, oth_pivot):
        ref_points = []
        for pivot in oth_pivot:
            dis = self.calculate_dis_arr(cluster, pivot)
            # print(dis)
            # print(self.Pos[-1])
            coeffs = self.train_model(dis, self.Pos)
            # print(coeffs[0] + sum(coeffs[i] * dis[0] ** i for i in range(1, Constants.COEFFS + 1)))
            # print(coeffs[0] + sum(coeffs[i] * dis[10000] ** i for i in range(1, Constants.COEFFS + 1)))
            # print(coeffs[0] + sum(coeffs[i] * dis[-1] ** i for i in range(1, Constants.COEFFS + 1)))
            # 创建 Ref_Point 实例
            ref_point = Ref_Point(point=pivot, r=max(dis), r_low=min(dis))
            # 使用设置方法分配 dis 和 coeffs
            ref_point.set_dis_arr(dis)
            ref_point.set_coeffs(coeffs)
            ref_point.set_model(coeffs)
            ref_points.append(ref_point)

        return ref_points

    def calculate_dis_arr(self, cluster, refPoint):
        dis = [np.linalg.norm(np.array(refPoint.coordinate) - np.array(p.coordinate)) for p in cluster]
        dis.sort()
        return dis

    ############################### test point 1 ####################################

    # def train_model(self, X, Y):
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #     poly_reg = PolynomialRegression()
    #     order = Constants.COEFFS
    #     coeffs = poly_reg.fit(X, Y, order)
    #     r2 = self.cacul_r2(Y, X, coeffs)
    #     #print(r2)
    #     # 绘制预测曲线和原始数据
    #     x_fit = X  # 平滑曲线
    #     y_fit = poly_reg.predict(x_fit, coeffs)

    #     # 按X值排序，因为fill_between要求x轴顺序
    #     sorted_indices = np.argsort(X)
    #     X_sorted = np.array(X)[sorted_indices]
    #     y_fit_sorted = y_fit[sorted_indices]
        
    #     # 设置画布大小
    #     plt.figure(figsize=(10, 6))

    #     # 绘制原始数据点
    #     plt.scatter(X, Y, label='原始数据点', color='skyblue', s=30, alpha=0.7, edgecolors='w', linewidth=0.1)

    #     # 绘制预测曲线
    #     plt.plot(x_fit, y_fit, color='red', label=f'预测曲线')


    #     # 移除坐标轴的刻度
    #     plt.xticks([])
    #     plt.yticks([])

    #     # 添加标题和坐标轴标签
    #     plt.title('多项式回归模型预测')
    #     plt.xlabel('距离')
    #     plt.ylabel('排名')

    #     # 添加图例
    #     plt.legend(loc='best')

    #     # 显示图形
    #     plt.tight_layout()
    #     plt.show()
    #     return coeffs

    # def train_model(self, X, Y):
    #     X_tensor = torch.FloatTensor(X).unsqueeze(1)
    #     Y_tensor = torch.FloatTensor(Y).unsqueeze(1)

    #     dataset = TensorDataset(X_tensor, Y_tensor)
    #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #     model = SimpleNN(input_dim=1)

    #     def custom_loss(outputs, targets):
    #         lower_bounds, upper_bounds = outputs[:, 0], outputs[:, 1]
    #         targets = targets.squeeze()

    #         # 确保下界小于等于上界
    #         constraint_loss = torch.mean(torch.relu(lower_bounds - upper_bounds))

    #         # 计算区间内外的误差
    #         in_bounds = (targets >= lower_bounds) & (targets <= upper_bounds)
    #         in_loss = torch.mean((upper_bounds - lower_bounds)[in_bounds])
    #         out_loss = torch.mean(torch.abs(targets - lower_bounds)[~in_bounds] + torch.abs(targets - upper_bounds)[~in_bounds])
    #         total_loss = in_loss + out_loss + constraint_loss

    #         return total_loss
        
    #     criterion = custom_loss  # 使用自定义的损失函数
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #     num_epochs = 30
    #     for epoch in range(num_epochs):
    #         print(epoch)
    #         for inputs, targets in dataloader:
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = criterion(outputs, targets)
    #             loss.backward()
    #             optimizer.step()

    #     # 在全部训练数据上评估模型并输出预测区间
    #     with torch.no_grad():
    #         X_tensor_eval = torch.FloatTensor(X).unsqueeze(1)
    #         predicted_intervals = model(X_tensor_eval).numpy()
    #         lower_bounds, upper_bounds = predicted_intervals[:, 0], predicted_intervals[:, 1]

    #     matplotlib.rcParams['font.size'] = 9
    #     matplotlib.rcParams['axes.labelsize'] = 10
    #     matplotlib.rcParams['axes.titlesize'] = 12

    #     # 绘制原始数据点和预测区间
    #     plt.figure(figsize=(10, 6))  # 设置画布大小

    #     # 确保X_array是排序后的，这样绘图时不会出现交叉线
    #     sorted_indices = np.argsort(X)
    #     X_sorted = np.array(X)[sorted_indices]
    #     lower_sorted = lower_bounds[sorted_indices]
    #     upper_sorted = upper_bounds[sorted_indices]

    #     def ensure_correct_order(lower, upper):
    #         # np.where返回满足条件的元素索引
    #         wrong_order = lower > upper
    #         lower[wrong_order], upper[wrong_order] = upper[wrong_order], lower[wrong_order]
    #         return lower, upper
        
    #     lower_sorted, upper_sorted = ensure_correct_order(lower_sorted, upper_sorted)

    #     delta = 500
    #     plt.plot(X_sorted, lower_sorted - (upper_sorted - lower_sorted) * 100, '--', color='gray', alpha=0.3, label='Lower Bound')
    #     plt.plot(X_sorted, upper_sorted + (upper_sorted - lower_sorted) * 100, '--', color='gray', alpha=0.3, label='Upper Bound')
    #     plt.plot(X_sorted, (upper_sorted + lower_sorted) / 2, color='red', alpha=1, label='Mean')
    #     plt.fill_between(X_sorted, lower_sorted - (upper_sorted - lower_sorted) * 100, upper_sorted + (upper_sorted - lower_sorted) * 100, where= upper_sorted >= lower_sorted, facecolor='lightgreen', alpha=0.1, interpolate=True, label='Prediction Interval')
    #     plt.scatter(X, Y, label='Original Data', color='skyblue', s=30, alpha=0.7, edgecolors='w', linewidth=0.1)

    #     # 添加网格线，提高可读性
    #     plt.grid(True, linestyle='--', alpha=0.5)

    #     # 设置标题和坐标轴标签
    #     plt.title('Neural Network Model Prediction with Confidence Interval')
    #     plt.xlabel('Distance')
    #     plt.ylabel('Rank')

    #     # 添加图例，使用最佳位置自动放置
    #     plt.legend(loc='best')

    #     # 调整整体空白使图像更加紧凑
    #     plt.tight_layout()

    #     # 显示图像
    #     plt.show()

    #     # 保存图像
    #     plt.savefig('prediction_interval.pdf', format='pdf', dpi=1200, bbox_inches='tight')

    #     return model

    ############################### test point 1 ####################################

#     def train_model(self, X, Y):
#             poly_reg = PolynomialRegression()
#             order = Constants.COEFFS
#             coeffs = poly_reg.fit(X, Y, order)
#             r2 = self.cacul_r2(Y, X, coeffs)
#             print(r2)
#             x_fit = X  # Smooth curve
#             y_fit = poly_reg.predict(x_fit, coeffs)
#             # plt.plot(x_fit, y_fit, color='red', label=f'Polynomial Degree {order}')
#             # plt.scatter(X, Y, label='Original Data')
#             # # Add title and labels
#             # plt.title('Polynomial Regression Fit')
#             # plt.xlabel('Independent variable')
#             # plt.ylabel('Dependent variable')

#             # # Show legend
#             # plt.legend()

#             # # Display the plot
#             # plt.show()
#             model_size = sys.getsizeof(coeffs)
#             print(f"Model size in bytes: {model_size}")
#             return coeffs

    # def train_model(self, X, Y):
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #     # 转换X和Y为torch张量
    #     X_tensor = torch.FloatTensor(X).unsqueeze(1)  # 假设X是一维的，需要增加一个维度以匹配网络输入
    #     Y_tensor = torch.FloatTensor(Y).unsqueeze(1)  # 同上

    #     dataset = TensorDataset(X_tensor, Y_tensor)
    #     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #     model = SimpleNN(input_dim=1, output_dim=1)
    #     criterion = torch.nn.MSELoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #     num_epochs = 1  # 设置适当的训练轮数
    #     for epoch in range(num_epochs):
    #         for inputs, targets in dataloader:
    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             loss = criterion(outputs, targets)
    #             loss.backward()
    #             optimizer.step()

    #     # 在全部训练数据上评估模型
    #     with torch.no_grad():
    #         X_tensor_eval = torch.FloatTensor(X).unsqueeze(1)
    #         predicted = model(X_tensor_eval).numpy()

    #     plt.figure(figsize=(10, 6))
    #     # 绘制原始数据点
    #     plt.scatter(X, Y, label='原始数据点', color='skyblue', s=30, alpha=0.7, edgecolors='w', linewidth=0.1)
        
    #     # 绘制模型预测结果
    #     plt.plot(X, predicted.flatten(), color='red', label='预测曲线', linewidth=2)

    #     # 添加标题和轴标签
    #     plt.title('神经网络模型预测', fontsize=12)
    #     plt.xlabel('距离', fontsize=10)
    #     plt.ylabel('排名', fontsize=10)
        
    #     # 移除坐标轴的刻度
    #     plt.xticks([])
    #     plt.yticks([])
    #     # 添加图例
    #     plt.legend(loc='best', fontsize=9)
        
    #     # 显示图形
    #     plt.tight_layout()
    #     plt.show()

    #     model_size_bytes = self.get_model_size(model)
    #     print(f"Model size in bytes: {model_size_bytes}")
    #     return model

    # def get_model_size(self, model):
    #     total_size_bytes = 0
    #     for param in model.parameters():
    #         # param.nelement() gives the total number of elements in the parameter,
    #         # param.element_size() gives the size in bytes of a single element
    #         total_size_bytes += param.nelement() * param.element_size()

    #     return total_size_bytes

    def cacul_r2(self, Y, X, coeffs):
        """
        计算多项式回归的R²值。
        """
        y_mean = sum(Y) / len(Y)  # 计算目标变量 Y 的平均值
        ss_tot = sum((y_i - y_mean) ** 2 for y_i in Y)  # 计算总平方和
        ss_res = sum((y_i - sum(coeffs[j] * x_i ** j for j in range(len(coeffs)))) ** 2 for x_i, y_i in zip(X, Y))  # 计算残差平方和
        r2 = 1 - (ss_res / ss_tot)  # 计算 R² 值
        return r2
    
    def caculate_circle_bound(self, dis):
        """
        计算给定距离列表的圆环边界。
        """
        dis.sort()
        circle_num = 30
        split = len(dis) // circle_num + 1
        dict_circle = []
        for i in range(circle_num):
            lower_bound = dis[i*split]
            upper_bound = dis[min((i+1)*split-1, len(dis)-1)]
            dict_circle.append([lower_bound, upper_bound])
        return dict_circle

    def caculate_r(self, cluster, point):
        """
        计算集合中所有点到某个特定点的最大距离。
        """
        r = 0.0
        for clu_point in cluster.clu_point:
            dis = euclidean(np.array(point.coordinate), np.array(clu_point.coordinate))
            r = max(r, dis)
        return r

    ############ test point 1-2 Linear spline function #########################

    # def train_model(self, X, Y):
    #     # 将数据转换为numpy数组
    #     X = np.array(X)
    #     Y = np.array(Y)
        
    #     # 确定knots的数量和位置
    #     knots = self.choose_knots(X, num_knots=10)
        
    #     # 创建线性样条模型，度数为1代表线性
    #     spline_model = LSQUnivariateSpline(X, Y, knots, k=1)
        
    #     # 绘图逻辑
    #     self.plot_spline(X, Y, spline_model, knots)
        
    #     # 返回样条模型
    #     return spline_model

    # def choose_knots(self, X, num_knots):
    #     # 生成knots位置
    #     return np.linspace(np.min(X), np.max(X), num_knots + 2)[1:-1]
    
    # def plot_spline(self, X, Y, spline_model, knots):
    #     # 用来正常显示中文标签
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
    #     # 设置画布大小
    #     plt.figure(figsize=(10, 6))
        
    #     # 绘制原始数据点
    #     plt.scatter(X, Y, label='原始数据点', color='skyblue', s=30, alpha=0.7, edgecolors='w', linewidth=0.1)
        
    #     # 绘制拟合的样条曲线
    #     xs = np.linspace(np.min(X), np.max(X), 1000)
    #     ys = spline_model(xs)
    #     plt.plot(xs, ys, label='线性样条曲线', color='red', linewidth=2)
        
    #     # 标记拐点位置
    #     plt.scatter(knots, spline_model(knots), marker='o', color='yellow', label='拐点', s=50, zorder=5)
        
    #     # 添加图例
    #     plt.legend(loc='best')
        
    #     # 添加标题和坐标轴标签
    #     plt.title('线性样条拟合')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
        
    #     # 显示网格
    #     plt.grid(True)
        
    #     # 显示图形
    #     plt.show()

    ######### AIC + Linear spline function #####################

    # def train_model(self, X, Y):
    #     # 将数据转换为numpy数组
    #     X = np.array(X)
    #     Y = np.array(Y)

    #     best_aic = np.inf
    #     best_model = None
    #     best_knots = None

    #     # 从1到最大可能的拐点数进行迭代
    #     for num_knots in range(1, min(len(X) - 2, 20)):
    #         # 生成knots位置
    #         knots = np.linspace(np.min(X), np.max(X), num_knots + 2)[1:-1]
    #         try:
    #             spline_model = LSQUnivariateSpline(X, Y, knots, k=1)
    #             rss = mean_squared_error(Y, spline_model(X)) * len(Y)
    #             # 计算AIC，k是参数的数量：系数+拐点
    #             aic = len(Y) * np.log(rss / len(Y)) + 2 * (num_knots + spline_model.get_coeffs().size)
    #             if aic < best_aic:
    #                 best_aic = aic
    #                 best_model = spline_model
    #                 best_knots = knots
    #         except Exception as e:
    #             # 忽略由于拐点选择不当导致的错误
    #             print("Error with knots:", knots, "->", e)

    #     # 绘图并返回最佳模型
    #     # self.plot_spline(X, Y, best_model, best_knots)

    #     model_size = self.get_model_size(best_model)
    #     print(f"Model size in bytes: {model_size}")
    #     # print(f"Best AIC: {best_aic}")
    #     return best_model

    def get_model_size(self, spline_model):
        # 计算系数数组的大小
        coeffs_size = spline_model.get_coeffs().nbytes
        # 计算节点数组的大小
        knots_size = spline_model.get_knots().nbytes

        # 计算总大小
        total_size = coeffs_size + knots_size
        return total_size


    # def plot_spline(self, X, Y, spline_model, knots):
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(X, Y, color='skyblue', label='原始数据点')
    #     xs = np.linspace(min(X), max(X), 1000)
    #     ys = spline_model(xs)
    #     plt.plot(xs, ys, color='red', label='预测曲线')
    #     plt.scatter(knots, spline_model(knots), color='yellow', marker='o', s=50, label='拐点')
    #     plt.title('AIC优化样条函数')
    #     plt.xlabel('距离')
    #     plt.ylabel('排名')

    #     # 移除坐标轴的刻度
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()


    # ############ 二阶导数Second derivative分段 ############
    
    # def train_model(self, X, Y):
    #     # 计算数据的二阶导数

    #     def second_derivative(x, y):
    #         n = len(y)
    #         second_deriv = np.zeros(n)
    #         for i in range(1, n-1):
    #             h1 = x[i] - x[i-1]
    #             h2 = x[i+1] - x[i]
    #             if h1 == 0 or h2 == 0:
    #                 continue  # 跳过因间隔为零导致的计算
    #             f1 = y[i-1]
    #             f2 = y[i]
    #             f3 = y[i+1]
    #             second_deriv[i] = 2 * ((f3 - f2) / h2 - (f2 - f1) / h1) / (h1 + h2)
    #         return second_deriv
    #     # 计算数据的二阶导数
    #     D2 = second_derivative(X, Y)
    #     # 识别分割点：二阶导数的极大值点，使用np.argwhere并处理输出格式
    #     split_indices = np.argwhere(D2 > np.percentile(D2, 99.9)).flatten()
        
    #     # 确保X是numpy数组以支持高级索引
    #     X = np.array(X)

    #     # 确保split_indices是整数数组
    #     split_indices = np.unique(split_indices).astype(int)  # 去重并确保为整数

    #     # 验证索引的有效性（确保不超出X的范围）
    #     split_indices = split_indices[split_indices < len(X)]

    #     # 生成拐点位置
    #     if len(split_indices) > 1:
    #         knots = X[split_indices]
    #     else:
    #         # 如果没有足够的分割点，返回一个默认的模型或处理方式
    #         knots = np.array([np.min(X), np.max(X)])  # 至少要有一个有效的knots

    #     try:
    #         spline_model = LSQUnivariateSpline(X, Y, knots, k=1)
    #     except Exception as e:
    #         best_aic = np.inf
    #         best_model = None
    #         best_knots = None

    #         # 从1到最大可能的拐点数进行迭代
    #         for num_knots in range(1, min(len(X) - 2, 20)):
    #             # 生成knots位置
    #             knots = np.linspace(np.min(X), np.max(X), num_knots + 2)[1:-1]
    #             try:
    #                 spline_model = LSQUnivariateSpline(X, Y, knots, k=1)
    #                 rss = mean_squared_error(Y, spline_model(X)) * len(Y)
    #                 # 计算AIC，k是参数的数量：系数+拐点
    #                 aic = len(Y) * np.log(rss / len(Y)) + 2 * (num_knots + spline_model.get_coeffs().size)
    #                 if aic < best_aic:
    #                     best_aic = aic
    #                     best_model = spline_model
    #                     best_knots = knots
    #             except Exception as e:
    #                 # 忽略由于拐点选择不当导致的错误
    #                 print("Error with knots:", knots, "->", e)

    #         spline_model = best_model

    #     # 绘图逻辑
    #     # self.plot_spline(X, Y, spline_model, knots)

    #     return spline_model


    
    # def plot_spline(self, X, Y, spline_model, knots):
    #     plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
    #     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(X, Y, color='skyblue', label='原始数据点')
    #     xs = np.linspace(min(X), max(X), 1000)
    #     ys = spline_model(xs)
    #     plt.plot(xs, ys, color='red', label='预测曲线')
    #     plt.scatter(knots, spline_model(knots), color='yellow', marker='o', s=50, label='拐点')
    #     plt.title('二阶导数优化样条函数')
    #     plt.xlabel('距离')
    #     plt.ylabel('排名')

    #     # 移除坐标轴的刻度
    #     plt.xticks([])
    #     plt.yticks([])

    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    #     matplotlib.rcParams['font.size'] = 9
    #     matplotlib.rcParams['axes.labelsize'] = 10
    #     matplotlib.rcParams['axes.titlesize'] = 12

    ######### AIC + two ###########



    def train_model(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)

        # print(f"Training model with {len(X)} data points.")

        # 检查数据点是否足够构建样条模型
        if len(X) < (100):
            # print(f"Insufficient data points ({len(X)}) for spline model. Returning simple linear regression model.")
            # 返回简单的线性回归模型
            model = LinearRegression().fit(X.reshape(-1, 1), Y)
            return model
    
        def calculate_second_derivative(x, y):
            """计算二阶导数"""
            n = len(y)
            second_deriv = np.zeros(n)
            for i in range(1, n - 1):
                h1 = x[i] - x[i - 1]
                h2 = x[i + 1] - x[i]
                if h1 == 0 or h2 == 0:
                    continue  # 避免除零错误
                f1, f2, f3 = y[i - 1], y[i], y[i + 1]
                second_deriv[i] = 2 * ((f3 - f2) / h2 - (f2 - f1) / h1) / (h1 + h2)
            # 仅在必要时打印摘要信息
            # print(f"Second derivative (summary): min={np.min(second_deriv)}, max={np.max(second_deriv)}")
            return second_deriv

        def initial_knots_selection(X, Y):
            """基于AIC选择初始拐点"""
            best_aic = np.inf
            best_knots = None
            print("Selecting initial knots...")
            for num_knots in range(1, min(20, len(X) - 2)):
                knots = np.linspace(np.min(X), np.max(X), num_knots + 2)[1:-1]
                try:
                    model = LSQUnivariateSpline(X, Y, t=knots, k=1)
                    rss = mean_squared_error(Y, model(X)) * len(Y)
                    aic = len(Y) * np.log(rss / len(Y)) + 2 * (num_knots + 1)
                    if aic < best_aic:
                        best_aic = aic
                        best_knots = knots
                except Exception as e:
                    # print(f"Error selecting knots with {num_knots} knots: {e}")
                    continue

            if best_knots is None:
                # print("No valid initial knots found. Using default knots.")
                best_knots = np.linspace(np.min(X), np.max(X), 3)[1:-1]  # 默认设置

            # print(f"Initial knots selected: {best_knots}")
            return best_knots

        def refine_knots_with_second_derivative(D2, X, initial_knots):
            """根据二阶导数优化拐点"""
            if initial_knots is None or len(initial_knots) == 0:
                # print("Initial knots are empty. Skipping refinement.")
                return initial_knots  # 返回初始拐点或跳过优化

            percentile_99_9 = np.percentile(np.abs(D2), 99.9)
            candidates = np.argwhere(D2 > percentile_99_9).flatten()

            max_knots = min(len(candidates), 10)
            top_candidate_indices = candidates[np.argsort(D2[candidates])[-max_knots:]]

            try:
                refined_knots = np.sort(np.unique(np.concatenate([X[top_candidate_indices], initial_knots])))
            except ValueError as ve:
                # print(f"Error refining knots: {ve}")
                refined_knots = initial_knots  # 回退到初始拐点

            # print(f"Refined knots: {refined_knots}")
            return refined_knots

        # 调用各个函数并打印必要的调试信息
        D2 = calculate_second_derivative(X, Y)
        initial_knots = initial_knots_selection(X, Y)
        refined_knots = refine_knots_with_second_derivative(D2, X, initial_knots)

        final_model = LSQUnivariateSpline(X, Y, t=refined_knots, k=1)
        model_size = self.get_model_size(final_model)
        print(f"Model size: {model_size} bytes")
        
#         plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
#         plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#         plt.figure(figsize=(10, 6))
#         plt.scatter(X, Y, color='skyblue', label='原始数据点')
#         xs = np.linspace(min(X), max(X), 1000)
#         ys = final_model(xs)
#         plt.plot(xs, ys, color='red', label='预测曲线')
#         plt.scatter(refined_knots, final_model(refined_knots), color='yellow', marker='o', s=50, label='拐点')
#         plt.title('AIC+二阶导数优化样条曲线')
#         plt.xlabel('距离')
#         plt.ylabel('排名')

#         # 移除坐标轴的刻度
#         plt.xticks([])
#         plt.yticks([])
#         plt.legend()
#         plt.grid(True)
#         plt.show()
        return final_model



