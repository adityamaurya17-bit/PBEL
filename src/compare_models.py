import matplotlib.pyplot as plt

# Accuracy values from your runs
log_reg_acc = 0.984
rf_acc = 0.978
xgb_acc = 0.979

models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracy = [log_reg_acc, rf_acc, xgb_acc]

plt.bar(models, accuracy, color=['skyblue','lightgreen','salmon'])
plt.ylim(0.95,1.0)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.show()