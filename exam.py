import numpy as np

class LineModel:
    def __init__(self):
        self.slope = 0
        self.bias = 0
        self.points = []

    def add_points(self, points):
        self.points = points

    def totat_distance(self):
        if not self.points:
            return 0
        x_coords, y_coords = zip(*self.points)
        m = self.slope
        c = self.bias
        return sum(abs(m*x - y + c) for x, y in zip(x_coords, y_coords))

    def line_equations(self):
        return f"y = {self.slope}x + {self.bias}"


class OptimizeLineModel(LineModel):

    def __init__(self, learning_rate=0.1, iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.iterations = iterations

    def OptimizeLine(self):
        if not self.points:
            return
        
        x_coords, y_coords = zip(*self.points)
        
        # Gradient Descent formula: m_i+1 = m_i - alpha * Σ (yi - (mxi + c))
        #                     c_i+1 = c_i - alpha * Σ (yi - (mxi + c))
        
        N = len(x_coords)
        
        m = 0.0
        c = 0.0
        
        for _ in range(self.iterations):
            m_new = 0.0
            c_new = 0.0
            
            for i in range(N):
                m_new += (y_coords[i] - m*x_coords[i] - c)
                c_new += (y_coords[i] - m*x_coords[i] - c)
            
            m -= self.learning_rate * m_new / N
            c -= self.learning_rate * c_new / N
            
            


# Usage:
points = [(1,2), (4,8), (3,6), (8,16)]
model = OptimizeLineModel(learning_rate=0.5, iterations=10)
model.add_points(points)
model.OptimizeLine()

print(f'Total distance: {model.totat_distance()}')
print(f'Line equation is: {model.line_equations()}')

