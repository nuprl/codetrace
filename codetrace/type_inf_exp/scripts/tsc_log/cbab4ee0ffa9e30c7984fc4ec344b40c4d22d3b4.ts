type __typ2 = string;

export const WILDCARD ='*';
export class ProxyFactory{
    _map:Map<any,__typ1> =new Map<any,__typ1>();

    builder(key:any):__typ1{
        if(!this._map.has(key)){
            this._map.set(key,new __typ1());
        }
        return this._map.get(key);
    }

    getBuilder(key:any):__typ1{
        var bld = this._map.get(key);
        if(!bld){
            bld=this._map.get(WILDCARD); 
        }
        return bld;
    }

}


export class __typ1{

    private _getters=new Map<__typ2,__typ5>();
    private _setters=new Map<__typ2,__typ6>();
    private _fns=new Map<__typ2,__typ9>();
    private createHandler():any{
        return {
            get:(obj: any,prop:__typ2)=>{
                var val;
               
                
                
                val= obj[prop];
                if(typeof(val)==='function'){
                    var fn = this._fns.get(prop);
                    if(!fn){
                        fn=this._fns.get(WILDCARD);
                    }
                    if(fn){
                        val=fn.proxyMethod(obj,prop,val);
                    } else{
                        return val.bind(obj);
                    }
                }
                else{   
                    var getter = this._getters.get(prop);
                    if(!getter)
                        getter=this._getters.get(WILDCARD); 
                    if(getter)
                        val= getter.invokeHandler(obj,prop);
                    else
                        val= obj[prop];
                }
              
                
                
                return val;
            },
            set:(obj:any,prop:__typ2,val:any)=>{
                var setter = this._setters.get(prop);
                if(!setter)
                    setter = this._setters.get(WILDCARD); 
                if(setter)
                    setter.invokeHandler(obj,prop,val);
                else
                    obj[prop]=val;
                return true; 
            }
        };
    }

    private isFn(obj:any){

    }

    
    isConfigured(){
        return this._fns.size || this._getters.size || this._setters.size;
    }

    proxy<__typ0 extends Object>(obj:__typ0){
        if(!this.isConfigured()){
            return obj; 
        }
        return new Proxy(obj,this.createHandler()) as __typ0;
    }

    get(prop:__typ2):__typ5{
        let gtr = this._getters.get(prop);
        if(!gtr){
            gtr = new __typ5();
            this._getters.set(prop,gtr);
        }
        return gtr;
    }
    set(prop:__typ2):__typ6{
        let str = this._setters.get(prop);
        if(!str){
            str = new __typ6();
            this._setters.set(prop,str);
        };
        return str;
    }
    fn(name:__typ2):__typ9{
        let fn = this._fns.get(name);
        if(!fn){
            fn=new __typ9();
            this._fns.set(name,fn);
        }
        return fn;
    }
}

interface __typ3{
    (target:any,prop:__typ2):any;
}

interface __typ4{
    (taget:any,prop:__typ2,value:any):any;
}

interface __typ8{
    (target:any,name:__typ2,args:any[]):any;
}

interface __typ7{
    (target:any,name:__typ2,args:any[],result:any):any;
}

export class __typ5{
    invokeHandler(target:any,prop:__typ2):any{
        
        let propValue=undefined;
        this._before.forEach(pi=>pi(target,prop));
        
        if(this._instead)
            propValue=this._instead(target,prop);
        else
            propValue=target[prop];
        
        this._after.forEach(pi=>{
            const aft = pi(target,prop,propValue);
            if(aft!==undefined){
                propValue=aft;
            }
        });
        return propValue;
    }
    
    private _before=new Array<__typ3>();
    private _instead:__typ3;
    private _after=new Array<__typ4>();
    before(fn:__typ3){
        this._before.push(fn);
    }
    after(fn:__typ4){
        this._after.push(fn);
    }
    instead(fn:__typ3){
        this._instead=fn;
    }
}

export class __typ6{
    invokeHandler(target:any,prop:__typ2,value:any):void{
        
        this._before.forEach(pi=>{
            const tmp =pi(target,prop,value)
            if(tmp!==undefined){
                value=tmp;
            }
        });
        
        if(this._instead)
            this._instead(target,prop,value);
        else
            target[prop]=value;
        
        this._after.forEach(pi=>{
            pi(target,prop,value);
        });
    }
    private _before=new Array<__typ4>();
    private _instead:__typ4;
    private _after=new Array<__typ4>();
    before(fn:__typ4){
        this._before.push(fn);
    }
    after(fn:__typ4){
        this._after.push(fn);
    }
    instead(fn:__typ4){
        this._instead=fn;
    }

    
}

export class __typ9{
        proxyMethod(obj:any,name:__typ2,fn:()=>any){
            return new Proxy(fn,{
                apply:(target,obj,args)=>{
                    var result=undefined;
                    this._before.forEach(f=>f(obj,name,args));
                    if(this._instead){
                        result = this._instead(obj,name,args);
                    }else{
                        result = fn.apply(obj,args);
                    }
                    this._after.forEach(f=>f(obj,name,args,result))
                    return result;
                }
            });
        }
        private _before=new Array<__typ8>();
        private _after=new Array<__typ7>();
        private _instead:__typ8;
        before(fn:__typ8){
            this._before.push(fn);
        }
        after(fn:__typ7){
            this._after.push(fn);
        }
        instead(fn:__typ8){
            this._instead=fn;
        }

}