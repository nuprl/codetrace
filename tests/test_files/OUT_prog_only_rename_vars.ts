export interface GrypeCvss {
    VendorMetadata: any;
    Metrics: Metrics;
    Vector: string;
    Version: string;
  }
  
  export interface Metrics {
    BaseScore: number;
    ExploitabilityScore: number;
    ImpactScore: number;
  }
  
  const __tmp29 = ['VendorMetadata', 'Metrics', 'Vector', 'Version'];
  
  export class Convert {
    public static toGrypeCvss(__tmp5: string): GrypeCvss[] {
      return __tmp34(JSON.parse(__tmp5), __tmp38(__tmp1('GrypeCvss')));
    }
  
    public static grypeCvssToJson(__tmp21: GrypeCvss[]): string {
      return JSON.stringify(__tmp13(__tmp21, __tmp38(__tmp1('GrypeCvss'))), null, 2);
    }

    public static grypeCvssToJson2(__tmp21: GrypeCvss): string {
      return JSON.stringify(__tmp13(__tmp21, __tmp1('GrypeCvss')), null, 2);
    }
  }
  
  function __tmp26(__tmp16: any, __tmp31: any, __tmp30: any = ''): never {
    if (__tmp30) {
      throw Error(`Invalid value for key "${__tmp30}". Expected type ${JSON.stringify(__tmp16)} but got ${JSON.stringify(__tmp31)}`);
    }
    throw Error(`Invalid value ${JSON.stringify(__tmp31)} for type ${JSON.stringify(__tmp16)}`);
  }
  
  function __tmp25(__tmp16: any): any {
    if (__tmp16.jsonToJS === undefined) {
      const __tmp36: any = {};
      __tmp16.props.forEach((__tmp3: any) => (__tmp36[__tmp3.json] = { key: __tmp3.js, typ: __tmp3.typ }));
      __tmp16.jsonToJS = __tmp36;
    }
    return __tmp16.jsonToJS;
  }
  
  function __tmp33(__tmp16: any): any {
    if (__tmp16.jsToJSON === undefined) {
      const __tmp36: any = {};
      __tmp16.props.forEach((__tmp3: any) => (__tmp36[__tmp3.js] = { key: __tmp3.json, typ: __tmp3.typ }));
      __tmp16.jsToJSON = __tmp36;
    }
    return __tmp16.jsToJSON;
  }
  
  function __tmp0(__tmp31: any, __tmp16: any, __tmp27: any, __tmp30: any = ''): any {
    function __tmp35(__tmp16: string, __tmp31: any): any {
      if (typeof __tmp16 === typeof __tmp31) return __tmp31;
      return __tmp26(__tmp16, __tmp31, __tmp30);
    }
  
    function __tmp11(__tmp24: any[], __tmp31: any): any {
      
      const __tmp37 = __tmp24.length;
      for (let __tmp6 = 0; __tmp6 < __tmp37; __tmp6++) {
        const __tmp16 = __tmp24[__tmp6];
        try {
          return __tmp0(__tmp31, __tmp16, __tmp27);
        } catch (_) {}
      }
      return __tmp26(__tmp24, __tmp31);
    }
  
    function __tmp20(__tmp22: string[], __tmp31: any): any {
      if (__tmp22.indexOf(__tmp31) !== -1) return __tmp31;
      return __tmp26(__tmp22, __tmp31);
    }
  
    function __tmp12(__tmp16: any, __tmp31: any): any {
      
      if (!Array.isArray(__tmp31)) return __tmp26('array', __tmp31);
      return __tmp31.map((__tmp2) => __tmp0(__tmp2, __tmp16, __tmp27));
    }
  
    function __tmp23(__tmp31: any): any {
      if (__tmp31 === null) {
        return null;
      }
      const __tmp7 = new Date(__tmp31);
      if (isNaN(__tmp7.valueOf())) {
        return __tmp26('Date', __tmp31);
      }
      return __tmp7;
    }
  
    function __tmp17(__tmp10: { [k: string]: any }, __tmp19: any, __tmp31: any): any {
      if (__tmp31 === null || typeof __tmp31 !== 'object' || Array.isArray(__tmp31)) {
        return __tmp26('object', __tmp31);
      }
      const __tmp18: any = {};
      Object.getOwnPropertyNames(__tmp10).forEach((__tmp30) => {
        const __tmp4 = __tmp10[__tmp30];
        const __tmp14 = Object.prototype.hasOwnProperty.call(__tmp31, __tmp30) ? __tmp31[__tmp30] : undefined;
        __tmp18[__tmp4.key] = __tmp0(__tmp14, __tmp4.typ, __tmp27, __tmp4.key);
      });
      Object.getOwnPropertyNames(__tmp31).forEach((__tmp30) => {
        if (!Object.prototype.hasOwnProperty.call(__tmp10, __tmp30)) {
          __tmp18[__tmp30] = __tmp0(__tmp31[__tmp30], __tmp19, __tmp27, __tmp30);
        }
      });
      return __tmp18;
    }
  
    if (__tmp16 === 'any') return __tmp31;
    if (__tmp16 === null) {
      if (__tmp31 === null) return __tmp31;
      return __tmp26(__tmp16, __tmp31);
    }
    if (__tmp16 === false) return __tmp26(__tmp16, __tmp31);
    while (typeof __tmp16 === 'object' && __tmp16.ref !== undefined) {
      __tmp16 = __tmp9[__tmp16.ref];
    }
    if (Array.isArray(__tmp16)) return __tmp20(__tmp16, __tmp31);
    if (typeof __tmp16 === 'object') {
      return __tmp16.hasOwnProperty('unionMembers')
        ? __tmp11(__tmp16.unionMembers, __tmp31)
        : __tmp16.hasOwnProperty('arrayItems')
        ? __tmp12(__tmp16.arrayItems, __tmp31)
        : __tmp16.hasOwnProperty('props')
        ? __tmp17(__tmp27(__tmp16), __tmp16.additional, __tmp31)
        : __tmp26(__tmp16, __tmp31);
    }
    
    if (__tmp16 === Date && typeof __tmp31 !== 'number') return __tmp23(__tmp31);
    return __tmp35(__tmp16, __tmp31);
  }
  
  function __tmp34<T>(__tmp31: any, __tmp16: any): T {
    return __tmp0(__tmp31, __tmp16, __tmp25);
  }
  
  function __tmp13<T>(__tmp31: T, __tmp16: any): any {
    return __tmp0(__tmp31, __tmp16, __tmp33);
  }
  
  function __tmp38(__tmp16: any) {
    return { arrayItems: __tmp16 };
  }
  
  function __tmp28(...__tmp24: any[]) {
    return { unionMembers: __tmp24 };
  }
  
  function __tmp32(__tmp10: any[], __tmp19: any) {
    return { props, additional }; // F: I will be filtering out shorthands
  }
  
  function __tmp8(__tmp19: any) {
    return { props: [], additional };
  }
  
  function __tmp1(__tmp15: string) {
    return { ref: __tmp15 };
  }
  
  const __tmp9: any = {
    GrypeCvss: __tmp32(
      [
        { json: 'VendorMetadata', js: 'VendorMetadata', typ: 'any' },
        { json: 'Metrics', js: 'Metrics', typ: __tmp1('Metrics') },
        { json: 'Vector', js: 'Vector', typ: '' },
        { json: 'Version', js: 'Version', typ: '' },
      ],
      false
    ),
    Metrics: __tmp32(
      [
        { json: 'BaseScore', js: 'BaseScore', typ: 3.14 },
        { json: 'ExploitabilityScore', js: 'ExploitabilityScore', typ: 3.14 },
        { json: 'ImpactScore', js: 'ImpactScore', typ: 3.14 },
      ],
      false
    ),
  };
  