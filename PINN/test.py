import numpy

# from PINN.lib.dataset.exact_1D_grid import T_MAX_PLOT

epochs = 20000
counter = 0

for epoch in range(epochs):
    if (epoch) % ((epochs) // 10) == 0:
        print(f"Epoch {epoch}/{epochs}") == 0
        counter += 1

# Dynamically select increasing amount of data points from the entire dataset during training. Should increase by 10% based on some threshold of the loss
# Should just slice from X_f, X_b, X_0 based on the counter variable and use that for training
        num_f = int(5000 * 10 * (counter / 10))
        num_b = int(500 * 10 * (counter / 10))
        num_0 = int(500 * 10 * (counter / 10))

        print(num_f, num_b, num_0)


