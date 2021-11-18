import matplotlib.pyplot as plt
 
num_list = [54.5, 57, 63, 68, 68.5, 69.5, 71, 71, 73.5, 76]
name_list = ['AlexNet', 'BN-AlexNet', 'BN-NIN', 'ENet', 'GoogleNet', 'ResNet18', 'VGG16', 'VGG19', 'ResNet34', 'ResNet50']

plt.figure('Networks Comparison')
plt.bar(range(len(num_list)), num_list, color=['r', 'tomato', 'gold', 'yellow', 'lawngreen', 'aqua', 'lightsteelblue', 'blue', 'violet', 'darkorchid'], tick_label=name_list)
plt.xticks(rotation=25)
plt.xlabel('Networks')
plt.ylabel('Top-1 Accuracy [%]')
plt.title('Performace of Different Networks')
plt.ylim((50, 80))
plt.grid(axis='y')
plt.show()