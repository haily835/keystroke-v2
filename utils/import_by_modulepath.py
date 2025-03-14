import importlib
from typing import Dict, Any

def import_by_modulepath(classpath):
    class_module = '.'.join(classpath.split('.')[:-1])
    class_name = classpath.split('.')[-1]
    
    module = importlib.__import__(
        class_module, 
        fromlist=[class_name]
    )

    args_class = getattr(module, class_name)
    return args_class



def initialize_class(classpath: str, init_args: Dict[str, Any] | None = None):
    class_module = '.'.join(classpath.split('.')[:-1])
    class_name = classpath.split('.')[-1]
    
    module = importlib.__import__(
        class_module, 
        fromlist=[class_name]
    )
    args_class = getattr(module, class_name)
    return args_class(**init_args) if init_args else args_class()