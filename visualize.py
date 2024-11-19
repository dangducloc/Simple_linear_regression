import matplotlib.pyplot as plt


class Visualize:
    
    def Visualzing(self,x_train,x_test,y_test,y_train,predicted_y_train ):
        # Plot the training data, testing data, and regression line
        plt.figure(figsize=(10, 6))
        plt.grid(visible = True)

        # Training data
        plt.scatter(x_train, y_train, color='blue',alpha = 0.2, label='Training Data')
        # Testing data
        plt.scatter(x_test, y_test, color='green',alpha = 0.2, label='Testing Data')

        # Plot the regression line (use both training and test x for a full line)
        plt.plot(x_train, predicted_y_train, color='red', label='Regression Line')

        # Add labels and title
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.title('Linear Regression: Price vs Area')
        plt.legend()

        # Show the plot
        plt.show()

