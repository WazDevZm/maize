export type RootTabParamList = {
  Home: undefined;
  Camera: undefined;
  History: undefined;
  Info: undefined;
};

export type RootStackParamList = {
  Main: undefined;
  DetectionResult: {
    result: DetectionResult;
    imageUri: string;
  };
  DiseaseDetail: {
    disease: DiseaseInfo;
  };
};