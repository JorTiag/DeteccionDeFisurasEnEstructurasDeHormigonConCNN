%Lectura de las carpetas.
%outputFolder=fullfile('SDNET2018');
rootFolder=fullfile('KaggleDataset');

%Distincion de categorias.
categorias={'ConFisura','SinFisura'};

IMDS=imageDatastore(fullfile(rootFolder,categorias),'LabeLSource','foldernames');

%Conteo de archivos de cada categoria ('Con Fisura, Sin Fisura').
tbl=countEachLabel(IMDS);

%Averigua que categoria contiene el menor número de archivos.
minSetCount=min(tbl{:,2})

%Iguala el numero de archivos en cada categoria (carpeta), tomando como 
%referencia el menor numero.
IMDS=splitEachLabel(IMDS,minSetCount,'randomize');
countEachLabel(IMDS);

%Leer las imagenes y etiquetar.
confisura=find(IMDS.Labels=='ConFisura',1);
sinfisura=find(IMDS.Labels=='SinFisura',1);

%Visualizacion de las imagenes.
figure
subplot(2,2,1);
imshow(readimage(IMDS,confisura));
title('Con fisura')
subplot(2,2,2);
imshow(readimage(IMDS,sinfisura));
title('Sin fisura')

%figure; montage({confisura,sinfisura},[]) 
%title('Con fisura Vs Sin fisura')

% Carga de ResNet-50
% Asignamos una variable (net) para almacenar la red.
net=resnet50();
figure
plot(net)
title('Arquitectura de ResNet-50')
%Dimenciona la figura de la arquitectura ResNet-50.
set(gca,'YLim',[150 170])

%Propiedades de la primera capa de entrada, avergua que tipo de entrada
%acepta.
net.Layers (1)
%Propiedades de la ultima capa (cantidad de clases de la red neuronal).
net.Layers(end)

%Otra forma de obtener la cantidad de clases de la red neuronal.
numel(net.Layers(end).ClassNames)

%Caracteristicas de la arquitectura ResNet-50.
analyzeNetwork(net)

%division para el entrenamiento y el test.
[trainingSet,testSet]=splitEachLabel(IMDS,0.7,'randomize');

%Obtener el tamaño requerido de la imagen de entrada.
imageSize=net.Layers(1).InputSize;

augmentedTrainingSet=augmentedImageDatastore(imageSize,trainingSet,'ColorPreprocessing','gray2rgb');
augmentedTestSet=augmentedImageDatastore(imageSize,testSet,'ColorPreprocessing','gray2rgb');

wl=net.Layers(2).Weights;
wl=mat2gray(wl);

%Primera convolución.
figure
montage(wl)
title('Capa de pesos de la primera convolución')

%Entrenamiento.
featureLayer = 'fc1000';
trainingFeatures = activations(net,augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLables = trainingSet.Labels;
classifier = fitcecoc (trainingFeatures, trainingLables,'Learner', 'Linear', 'Coding', 'onevsall','ObservationsIn','columns');

%Validación (test).
testFeatures = activations(net,augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

predictLabels=predict(classifier,testFeatures,'ObservationsIn','columns');

%Matriz de confusión.
testLables=testSet.Labels
confusionmat(testLables,predictLabels);

%Exactitud
accuracy = sum(predictLabels == testLables)/numel(testLables)
