"""Independent owned-array lower bounds, excluding Python/runtime/context overhead."""
import dataclasses
import types
import numpy as np

def host_array_bytes(*roots):
    """Count unique allocation bases reachable from supplied session/forcing objects."""
    seen=set();arrays={}
    def visit(obj):
        if obj is None or isinstance(obj,(str,bytes,int,float,bool,complex,type,types.ModuleType,
                                          types.FunctionType,types.MethodType)):return
        ident=id(obj)
        if ident in seen:return
        seen.add(ident)
        if isinstance(obj,np.ndarray):
            base=obj
            while isinstance(base.base,np.ndarray):base=base.base
            if base.size:
                pointer=int(base.__array_interface__['data'][0]);arrays[pointer]=max(arrays.get(pointer,0),base.nbytes)
            return
        if hasattr(obj,'__cuda_array_interface__'):return
        if isinstance(obj,dict):
            for value in obj.values():visit(value)
        elif isinstance(obj,(list,tuple,set)):
            for value in obj:visit(value)
        elif dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj):visit(getattr(obj,field.name))
        elif type(obj).__module__.startswith(('leaf.','maple_syrup.')):
            for value in getattr(obj,'__dict__',{}).values():visit(value)
            for slot in getattr(obj,'__slots__',()):visit(getattr(obj,slot,None))
    for obj in roots:visit(obj)
    return sum(arrays.values())
