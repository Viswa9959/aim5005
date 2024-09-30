import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully! 
        return x-self.minimum/(self.maximum-self.minimum)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def _check_is_array(self, x: Union[List, np.ndarray]) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    def fit(self, x: Union[List, np.ndarray]) -> None:   
        x = self._check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        
    def transform(self, x: Union[List, np.ndarray]) -> np.ndarray:
        """
        Standardize the given vector
        """
        x = self._check_is_array(x)
        return (x - self.mean) / self.std
    
    def fit_transform(self, x: List) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index: Dict[Any, int] = {}
        
    def _check_is_array(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Try to convert y to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(y, np.ndarray):
            y = np.array(y)
            
        assert isinstance(y, np.ndarray), "Expected the input to be a list"
        return y
        
    def fit(self, y: Union[List, np.ndarray]) -> None:
        """
        Fit the LabelEncoder to the given labels
        """
        y = self._check_is_array(y)
        unique_classes = sorted(set(y))
        self.classes_ = np.array(unique_classes)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        
    def transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Transform labels to normalized encoding
        """
        y = self._check_is_array(y)
        return np.array([self.class_to_index[item] for item in y])
    
    def fit_transform(self, y: Union[List, np.ndarray]) -> np.ndarray:
        """
        Fit the LabelEncoder and transform the given labels
        """
        self.fit(y)
        return self.transform(y)
