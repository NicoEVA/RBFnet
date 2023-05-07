# evaluate the NN at a higher resolution!

n = 100
eval_x = np.linspace(0,4,n)
activations_hr = np.zeros((5, n))
for i, center in enumerate(centers):
    activations_hr[i, :] = np.exp(-0.5*(eval_x-centers[i])**2)*w[:,i]
    plt.plot(eval_x, activations_hr[i, :], "--")
print(activations_hr)


reconstruct_hr_full = np.sum(activations_hr, axis=0)
print(reconstruct_hr_full)

y_hr = np.sin(eval_x)

plt.plot(eval_x, y_hr, "green")
plt.plot(eval_x, reconstruct_hr_full, "black")
plt.scatter(x, y, edgecolors="red")
