import random
import math
import matplotlib.pyplot as plt
import csv

class RandomPoints:
    '''A class to generate random points around given centers.'''

    def __init__(self, total_points=1000):
        self.total_points = total_points
        self.centers = [(0, 0), (1, 2), (2, 0)]
        self.generated_points = set()  # Para asegurar que no haya puntos repetidos
        self.x_values = []
        self.y_values = []

    def generate_points(self):
        '''Generate random points around the specified centers.'''
        # Calcular puntos por centro
        points_per_center = self.total_points // len(self.centers)
        extra_points = self.total_points % len(self.centers)

        for center in self.centers:
            num_points = points_per_center + (1 if extra_points > 0 else 0)
            extra_points -= 1
            while len([p for p in self.generated_points if center[0] - 1 <= p[0] <= center[0] + 1 and center[1] - 1 <= p[1] <= center[1] + 1]) < num_points:
                # Generate a random angle and distance within the radius
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(0, 1)
                # Calculate the new point's coordinates
                x = center[0] + distance * math.cos(angle)
                y = center[1] + distance * math.sin(angle)
                # Round the coordinates to a reasonable precision to avoid floating point issues
                x = round(x, 5)
                y = round(y, 5)
                # Ensure the point is not exactly at the center and is unique
                if (x, y) != center and (x, y) not in self.generated_points:
                    self.x_values.append(x)
                    self.y_values.append(y)
                    self.generated_points.add((x, y))

# Generar los puntos
random_points = RandomPoints(total_points=1000)
random_points.generate_points()

# Graficar los puntos generados
plt.style.use('seaborn-v0_8-dark')
fig, ax = plt.subplots()
scatter = ax.scatter(random_points.x_values, random_points.y_values, color='blue', s=10)

# Títulos y etiquetas
ax.set_title('Random Points around Centers', fontsize=14)
ax.set_xlabel('X Coordinate', fontsize=14)
ax.set_ylabel('Y Coordinate', fontsize=14)

# Mostrar el gráfico
plt.show()
plt.savefig('plot.png')

with open('points.csv', mode='w', newline="") as file:
    random_points = RandomPoints()
    random_points.generate_points()
    writer = csv.writer(file)
    writer.writerow(['Coordinate X', 'Coordinate Y'])
    for i in range(len(random_points.x_values)):
        writer.writerow([random_points.x_values[i],random_points.y_values[i]])

