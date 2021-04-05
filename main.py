import numpy as np
import itertools
import copy
import matplotlib.pyplot as plt
from time import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GA:
    def __init__(self, M, T, pCorss, pMutate):
        self.pos = []
        self.M = M  # 种群规模
        self.T = T  # 运行代数
        self.t = 0
        self.pCorss = pCorss  # 交叉概率
        self.pMutate = pMutate  # 变异概率
        self.cityNum = 0  # 城市数量，染色体长度
        self.distance = []  # 距离矩阵
        self.bestDistance = 0  # 最佳长度
        self.bestPath = []  # 最佳路径
        self.oldPopulation = []  # 父代种群
        self.newPopulation = []  # 子代种群
        self.fitness = []  # 个体的适应度
        self.Pi = []  # 个体的累积概率
        self.record = []  # 记录适应度变化

        self.ReadFile("data/datasets.txt")
        self.InitDist()  # 初始化距离矩阵
        self.InitPopulation()  # 初始化种群
        self.UpdateFitness()  # 初始化适应度
        self.CountRate()  # 初始化累计概率

    # 读取文件
    def ReadFile(self, filepath):
        infile = open(filepath)
        for line in infile:
            linedata = line.strip().split()
            self.pos.append([float(linedata[0]), float(linedata[1])])
        infile.close()
        self.cityNum = len(self.pos)  # 城市数量，染色体长度
        self.distance = np.zeros([self.cityNum, self.cityNum], dtype=int)

    # 初始化dist矩阵
    def InitDist(self):
        outfile = open("edge.txt", 'w')
        for i in range(self.cityNum):
            for j in range(i, self.cityNum):
                self.distance[i][j] = self.distance[j][i] = self.Distance(self.pos[i], self.pos[j])
                if i != j:
                    outfile.write(str(i) + "  " + str(j) + "  " + str(self.distance[i][j]) + "\n")

    # 随机初始化种群
    def InitPopulation(self):
        # 初始化种群
        for k in range(self.M):
            tmp = np.arange(self.cityNum)
            np.random.shuffle(tmp)
            self.oldPopulation.append(tmp)

    # 更新个体适应度
    def UpdateFitness(self):
        self.fitness.clear()
        for i in range(self.M):
            self.fitness.append(self.Evaluate(self.oldPopulation[i]))
        self.record.append(np.sum(self.fitness) / self.M)

    # 计算某个染色体的实际距离作为染色体适应度
    def Evaluate(self, chromosome):
        len = 0
        for i in range(1, self.cityNum):
            len += self.distance[chromosome[i - 1]][chromosome[i]]
        len += self.distance[chromosome[0]][chromosome[self.cityNum - 1]]  # 回到起点
        return len

    # 计算欧氏距离矩阵
    def Distance(self, pos1, pos2):
        return np.around(np.sqrt(np.sum(np.power(np.array(pos1) - np.array(pos2), 2))))
    # 适应度转化函数
    def FitFunc(self, fit):
        return 10000 / fit

    # 计算种群中每个个体的累积概率
    def CountRate(self):
        tmpFit = self.FitFunc(np.array(self.fitness))
        sum = np.sum(tmpFit)
        self.Pi = tmpFit / sum
        self.Pi = list(itertools.accumulate(self.Pi))
        self.Pi[self.M - 1] = np.round(self.Pi[self.M - 1]) #最后四舍五入保证累计概率的最后一个值为1

    # 轮盘挑选子代个体
    def SelectChild(self):
        self.newPopulation.clear()
        # 保留适应度最高的个体
        bestId = np.argmin(self.fitness)
        self.bestPath = copy.deepcopy(self.oldPopulation[bestId])
        self.bestDistance = self.fitness[bestId]
        self.newPopulation.append(copy.deepcopy(self.bestPath))
        for i in range(1, self.M):
            rate = np.random.random(1)
            for oldId in range(self.M):
                if self.Pi[oldId] >= rate:
                    self.newPopulation.append(copy.deepcopy(self.oldPopulation[oldId]))
                    break

    # 进化种群
    def Evolution(self):
        self.SelectChild()  # 选择
        rand = np.arange(self.M)
        np.random.shuffle(rand)
        for k in range(1, self.M, 2):
            rateC = np.random.random(1)
            if rateC < self.pCorss:  # 交叉
                self.OrderCross(rand[k], rand[k - 1])
            rateM = np.random.random(1)
            if rateM < self.pMutate:  # 变异
                self.Variation(rand[k])
            rateM = np.random.random(1)
            if rateM < self.pMutate:
                self.Variation(rand[k - 1])

    # 产生2个索引，用于交叉和变异
    def RandomRange(self):
        left = 0
        right = 0
        while left == right:
            left = np.random.randint(0, self.cityNum)
            right = np.random.randint(0, self.cityNum)
        ran = np.sort([left, right])
        left = ran[0]
        right = ran[1]
        return (left, right)

    # 变异算子
    def Variation(self, k):
        ran = self.RandomRange()
        left = ran[0]
        right = ran[1]
        while left < right:
            tmp = self.newPopulation[k][left]
            self.newPopulation[k][left] = self.newPopulation[k][right]
            self.newPopulation[k][right] = tmp
            left = left + 1
            right = right - 1

    # 顺序交叉算子
    def OrderCross(self, k1, k2):
        child1 = []
        child2 = []
        ran = self.RandomRange()
        left = ran[0]
        right = ran[1]

        swap1 = self.newPopulation[k1][left:right + 1]
        swap2 = self.newPopulation[k2][left:right + 1]
        i = 0
        while i < len(self.newPopulation[k2]):
            if len(child1) == left:
                child1.extend(swap1)
            elif self.newPopulation[k2][i] not in swap1:
                child1.append(self.newPopulation[k2][i])
            i = i + 1
        i = 0
        while i < len(self.newPopulation[k1]):
            if len(child2) == left:
                child2.extend(swap2)
            elif self.newPopulation[k1][i] not in swap2:
                child2.append(self.newPopulation[k1][i])
            i = i + 1

    # 开始GA
    def Run(self):
        count = 0
        for self.t in range(self.T):
            self.Evolution()
            self.oldPopulation.clear()
            self.oldPopulation = copy.deepcopy(self.newPopulation)
            self.UpdateFitness()
            self.CountRate()
            self.t = self.t + 1
            if self.record[self.t] == self.record[self.t - 1]:
                count = count + 1
            if count > 3:
                break

        print("结果:")
        print(self.bestPath)
        print(self.bestDistance)

    # 显示结果
    def Show(self):
        plt.title('TSP-GA')
        ax1 = plt.subplot(221)
        ax1.set_title('原始坐标')
        ax1.set_xlabel('x坐标')
        ax1.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')

        ax2 = plt.subplot(222)
        ax2.set_title('适应度变化')
        ax2.set_xlabel('代数')
        ax2.set_ylabel('平均代价')
        for i in range(1, len(self.record)):
            plt.plot([i, i - 1], [self.record[i], self.record[i - 1]], marker='o', color='k', markersize='1')

        ax3 = plt.subplot(223)
        ax3.set_title('线路')
        ax3.set_xlabel('x坐标')
        ax3.set_ylabel('y坐标')
        for point in self.pos:
            plt.plot(point[0], point[1], marker='o', color='k')
        for i in range(1, self.cityNum):
            plt.plot([self.pos[self.bestPath[i]][0], self.pos[self.bestPath[i - 1]][0]],
                     [self.pos[self.bestPath[i]][1], self.pos[self.bestPath[i - 1]][1]], color='g')
        plt.plot([self.pos[self.bestPath[0]][0], self.pos[self.bestPath[self.cityNum - 1]][0]],
                 [self.pos[self.bestPath[0]][1], self.pos[self.bestPath[self.cityNum - 1]][1]], color='g')

        plt.show()

    def MST(self):
        self.bestPath = [10, 11, 14, 17, 6, 27, 16, 26, 18, 5, 29, 23, 9, 25, 3, 1, 28, 4, 15, 21, 2, 8, 0, 7, 19, 20,
                         12, 24, 13, 22]
        self.bestDistance = self.Evaluate(self.bestPath)
        print(self.bestPath)
        print(self.bestDistance)


def main():
    t1=time()
    ga = GA(M=100, T=1000, pCorss=0.7, pMutate=0.3)
    ga.Run()
    t2=time()
    print("耗时:"+str(t2-t1))
    ga.Show()


if __name__ == '__main__':
    main()
