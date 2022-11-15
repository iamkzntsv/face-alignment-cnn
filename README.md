# Face Alignment using Convolutional Neural Network

Facial landmark detection acts as a fundamental part of many computer vision applications. It is commonly used for tasks such as facial expression analysis and real-time face tracking, so it needs to be as accurate and fast as possible. Common approaches[^1][^2] suggest training a cascade of regression trees using gradient boosting algorithm and pixel intensity differences as features. Alternatively, this problem can be approached using neural networks, which is the focus of this project. In addition, we will discuss how the detected landmarks can be used for lip and eye colour modification.

## Methods

### Data Augmentation

The amount of data in the original dataset may not be enough for proper model generalization. To overcome this problem several data augmentation techniques have been implemented. First, each image is reflected horizontally along with the corresponding landmarks. Random rotations and noise are then applied, and finally brightness shifts and vignette are added. As a result, we get a dataset of 16866 images, which we can now use to train the model.

### Pre-processing

Before passing our data to the neural network, we need to consider a few techniques to make it more consistent. Since optimizers are sensitive to the feature scale, we need to normalize pixel values so that they end up in the range $0$ to $1$, which can be done by dividing the value of each pixel by $255$. Also, we don't need our images to have a high resolution and we can compress them to a lower one $(96x96)$ to speed up training.

### Facial Landmark Detection

### Lip/Eye Colour Modification

## Experimental Results

## Conclusion
We have demonstrated how the face alignment problem can be solved with a CNN and compared different network architectures. We found that the pre-trained Inception model with trainable weights performs the best, although sometimes it still makes incorrect predictions. Finally, we looked at how predicted landmarks can be used to change lip and eye color. However, the accuracy of these operations is highly dependent on the quality of the landmarks and external factors such as lighting conditions and person's face morphology.

This report discussed one possible approach for predicting facial landmarks. The results show that transfer learning using the Inception model is an efficient and reliable method for facial landmark detection. It does not claim to be the best solution to the problem, but it can provide a relatively robust framework for achieving good results.

[^1]: Xudong Cao, Yichen Wei, Fang Wen, and Jian Sun. Face alignment by explicit shape regression. Inter-
national journal of computer vision, 107(2):177–190, 2014.
[^2]: Vahid Kazemi and Josephine Sullivan. One millisecond face alignment with an ensemble of regression
trees. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1867–
1874, 2014.

