import matplotlib.pyplot as plt

from Models.rvGA_model.run_ga_model import max_fitness_values, mean_fitness_values


plt.plot(max_fitness_values, color="red")
plt.plot(mean_fitness_values, color="green")
plt.xlabel("Generation")
plt.ylabel("Max/Mean fitness")
plt.xlim(0, 50)
plt.ylim(50, 101)
plt.title("Dependence of max/mean fitness on the number of generations")
plt.show()
