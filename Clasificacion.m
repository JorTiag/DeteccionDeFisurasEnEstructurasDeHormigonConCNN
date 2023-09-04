%clc   %Limpiar command window.
%clear all  %Borrar variables de Workspace
%close all  %Cerrar figuras abiertas

%% Datos del canal: 
ChannelIDFisuras = 2227566;
readAPIKeyFisuras = '9U91DTUCZJ2LGNQG';
writeAPIKeyFisuras = '2EPJ3ZEPKYRNPCIF';
    
%% Inicio captura de datos durante el tiempo de espera.
FisuraDetectada = 0;
NumeroFisuras = 0;
AreaFisura = 0;
NumImgFisura = 0;
NumImgSFisura = 0;
% thingSpeakWrite(ChannelIDFisuras,[FisuraDetectada,NumeroFisuras,AreaFisura,NumImgFisura,NumImgSFisura],'Writekey',writeAPIKeyFisuras);
% % Hay que esperar 15 segundos para dar tiempo a la escritura.
% pause(2); %Solo se ademiten esperas de 2 segundos
% pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); 
    
%%Obtener ruta actual.
ruta=pwd;

%%Agregar carpetas y subcarpetas donde estan imagenes.
addpath(genpath('ModeloKaggle'));

%Lectura de las carpetas.
outputFolde=fullfile('FotosAnalisisImagen');
rootFolde=fullfile(outputFolde,'EdificacionObraCivil');

%Distinción de categorias.
categoria={'ArchivoSantaCruzdeTenerife','CARResudenciaBlume','CasaCulturaTenerife','CasonRetiro','CorreaCubiertaZaragoza','FrayBernardino','MuseoAduanaMalaga','MuseoRomanticismo','ParquedeBomberos','Almonte','RioEspana','Toledo','Valmojado'};

IMD=imageDatastore(fullfile(rootFolde,categoria),'LabeLSource','foldernames');

%conteo de archivos
tbl=countEachLabel(IMD)
%minCount=min(tbl{:,2})
%maxCount=max(tbl{:,2})
TotalFotos=sum(tbl{:,2})

%%Colocarnos en la carpeta donde están las imagenes
cd HormigonPruebas

%%El tipo de archivo que acepta es jpg.
%dirImagen1 = dir('**/*.jpg'); %Busca todos los archivos en la carpeta y subcarpeta
dirImagen1 = dir('*.jpg');  %Busca todos los archivos en la carpeta
numimagenes1= length(dirImagen1);
datos = cell(1, numimagenes1);

ContImg=0;
DatosCategoria='Categoria';
c=0;
ConF=0;
s=0;
A=0;
AreaT=0;
NumF = 0;
NumFisT = 0;

%% Detección y analisis de las imagenes con fisura y sin fisura.
for k = 1:numimagenes1
    
    %%%Para imagenes colocar imread.
    datos{1,k} = imread(dirImagen1(k).name);
    ContImg=[ContImg;k];
    Foto=k;
    %figure(k);
    %imshow(datos{1,k});

    %%Ejecuta el modelo, clasifica las imagenes.
    ds=augmentedImageDatastore(imageSize,datos{1,k},'ColorPreprocessing','gray2rgb');
    imageFeatures = activations(net,ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');
    label=predict(classifier,imageFeatures,'ObservationsIn','columns');
    sprintf('La imagen %d pertenece a la categoria %s',Foto,label)

    DatosCategoria=[DatosCategoria;label];

    %Conteo de imagenes ConFisura y Sin fisura.
    if label == 'ConFisura'
        c=c+1;
        ConF=[ConF;k];
        
        %Filtro imfilter.
        h = fspecial('log',7,0.4);
        I2 = imfilter(datos{1,k},h);
        I2 = rgb2gray(I2);            %función rgb2gray
        figure (k); montage({datos{1,k},I2},[])
        %title('Original Vs Filtro imfilter')
        %title('Original Vs Escala de Grises')
       
        %Filtro para suavizar el ruido.
        I2 = wiener2(I2, [50 50]);
        %figure (k); montage({datos{1,k},I2},[])
        %title('Original Vs Filtro wiener2')

        %Binarizacion de la imagen.
        I2=im2bw(I2,0.40);
        %figure (k); montage({datos{1,k},I2},[])
        %title('Original Vs Función im2bw')
         
        % Relleno de gaps.
        se=strel('disk',3);
        I2=imclose(I2,se);
        %figure (k); montage({datos{1,k},I2},[])
        %title('Original Vs Relleno de Gaps')
         
        %Eliminacion de ruido (Elimina las partes que tengan mas de 130 megapixeles)
        I2=bwareaopen(I2,6);
        %figure (k); montage({datos{1,k},I2},[])
        %title('Original Vs Filtro bwareaopen')

        % Segmentacion.
        F = bwlabel(I2);
        
        %Propiedades de la fisura (datos de la segmentacion).
        PropF = regionprops(F, 'all');
        
        %Numero de fisuras de la imagen.
        NumF = max(max(F));
        NumFisT = [NumFisT;NumF];
        disp('Número de Fisuras:'), disp(NumF)
        
        %Area de las fisuras.
        A = sum([PropF.Area])
        AreaT = [AreaT;A];
        
        %figure (k); montage({datos{1,k},label2rgb(F)},[])
        %title('Original Vs Segmentación')
        
        % Visualización de la imagen original, filtrada y segmentada.
        figure (k); montage({datos{1,k},I2,label2rgb(F)},[])
        title('Original Vs Filtrada, Segmentada')
        
        %% Captura de datos de cada imagen.
        FisuraDetectada = k;
        NumeroFisuras = NumF;
        AreaFisura = A;
        NumImgFisura = c;
        NumImgSFisura = s;
%         thingSpeakWrite(ChannelIDFisuras,[FisuraDetectada,NumeroFisuras,AreaFisura,NumImgFisura,NumImgSFisura],'Writekey',writeAPIKeyFisuras);
%         % Hay que esperar 15 segundos para dar tiempo a la escritura.
%         pause(2); %Sólo se ademiten esperas de 2 segundos
%         pause(2); pause(2); pause(2); pause(2); pause(2); pause(2); pause(2);
    
    else
        
        s=s+1;
       
    end
    
end

%Tablas de datos.
TablaCategorias=table(ContImg,DatosCategoria);
TablaConteo=table(c,s);
TablaPropiedadesFisura=table(ConF,NumFisT,AreaT);

%%Regresar a ruta principal.
cd (ruta);

%%Exportar datos a exel
writetable(TablaCategorias,'DatosCategorias.xls','Sheet',2,'Range','B2');
writetable(TablaConteo,'DatosCategorias.xls','Sheet',2,'Range','E2');
writetable(TablaPropiedadesFisura,'DatosFisura.xls','Sheet',1,'Range','B2');