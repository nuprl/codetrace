type __typ0 = number;
type __typ4 = string;

export interface __typ2 {
    VendorMetadata: any;
    Metrics: __typ1;
    Vector: __typ4;
    Version: __typ4;
  }
  
  export interface __typ1 {
    BaseScore: __typ0;
    ExploitabilityScore: __typ0;
    ImpactScore: __typ0;
  }
  
  const names = ['VendorMetadata', 'Metrics', 'Vector', 'Version'];
  
  export class Convert {
    public static toGrypeCvss(__tmp1: __typ4): __typ2[] {
      return cast(JSON.parse(__tmp1), a(r('GrypeCvss')));
    }
  
    public static grypeCvssToJson(value): __typ4 {
      return JSON.stringify(uncast(value, a(r('GrypeCvss'))), null, 2);
    }

    public static grypeCvssToJson2(value) {
      return JSON.stringify(uncast(value, r('GrypeCvss')), null, 2);
    }
  }
  
  function invalidValue(__tmp3: any, __tmp4: any, key: any = '') {
    if (key) {
      throw Error(`Invalid value for key "${key}". Expected type ${JSON.stringify(__tmp3)} but got ${JSON.stringify(__tmp4)}`);
    }
    throw Error(`Invalid value ${JSON.stringify(__tmp4)} for type ${JSON.stringify(__tmp3)}`);
  }
  
  function jsonToJSProps(__tmp3: any): any {
    if (__tmp3.jsonToJS === undefined) {
      const map: any = {};
      __tmp3.props.forEach((p: any) => (map[p.json] = { key: p.js, typ: p.typ }));
      __tmp3.jsonToJS = map;
    }
    return __tmp3.jsonToJS;
  }
  
  function jsToJSONProps(__tmp3: any): any {
    if (__tmp3.jsToJSON === undefined) {
      const map: any = {};
      __tmp3.props.forEach((p: any) => (map[p.js] = { key: p.json, typ: p.typ }));
      __tmp3.jsToJSON = map;
    }
    return __tmp3.jsToJSON;
  }
  
  function transform(__tmp4: any, __tmp3: any, getProps: any, key: any = ''): any {
    function transformPrimitive(__tmp3: __typ4, __tmp4: any): any {
      if (typeof __tmp3 === typeof __tmp4) return __tmp4;
      return invalidValue(__tmp3, __tmp4, key);
    }
  
    function transformUnion(typs, __tmp4: any): any {
      
      const l = typs.length;
      for (let i = 0; i < l; i++) {
        const __tmp3 = typs[i];
        try {
          return transform(__tmp4, __tmp3, getProps);
        } catch (_) {}
      }
      return invalidValue(typs, __tmp4);
    }
  
    function transformEnum(cases: __typ4[], __tmp4: any): any {
      if (cases.indexOf(__tmp4) !== -1) return __tmp4;
      return invalidValue(cases, __tmp4);
    }
  
    function transformArray(__tmp3: any, __tmp4: any): any {
      
      if (!Array.isArray(__tmp4)) return invalidValue('array', __tmp4);
      return __tmp4.map((el) => transform(el, __tmp3, getProps));
    }
  
    function transformDate(__tmp4: any): any {
      if (__tmp4 === null) {
        return null;
      }
      const d = new Date(__tmp4);
      if (isNaN(d.valueOf())) {
        return invalidValue('Date', __tmp4);
      }
      return d;
    }
  
    function transformObject(__tmp2: { [k: __typ4]: any }, additional: any, __tmp4: any): any {
      if (__tmp4 === null || typeof __tmp4 !== 'object' || Array.isArray(__tmp4)) {
        return invalidValue('object', __tmp4);
      }
      const result: any = {};
      Object.getOwnPropertyNames(__tmp2).forEach((key) => {
        const prop = __tmp2[key];
        const v = Object.prototype.hasOwnProperty.call(__tmp4, key) ? __tmp4[key] : undefined;
        result[prop.key] = transform(v, prop.typ, getProps, prop.key);
      });
      Object.getOwnPropertyNames(__tmp4).forEach((key) => {
        if (!Object.prototype.hasOwnProperty.call(__tmp2, key)) {
          result[key] = transform(__tmp4[key], additional, getProps, key);
        }
      });
      return result;
    }
  
    if (__tmp3 === 'any') return __tmp4;
    if (__tmp3 === null) {
      if (__tmp4 === null) return __tmp4;
      return invalidValue(__tmp3, __tmp4);
    }
    if (__tmp3 === false) return invalidValue(__tmp3, __tmp4);
    while (typeof __tmp3 === 'object' && __tmp3.ref !== undefined) {
      __tmp3 = typeMap[__tmp3.ref];
    }
    if (Array.isArray(__tmp3)) return transformEnum(__tmp3, __tmp4);
    if (typeof __tmp3 === 'object') {
      return __tmp3.hasOwnProperty('unionMembers')
        ? transformUnion(__tmp3.unionMembers, __tmp4)
        : __tmp3.hasOwnProperty('arrayItems')
        ? transformArray(__tmp3.arrayItems, __tmp4)
        : __tmp3.hasOwnProperty('props')
        ? transformObject(getProps(__tmp3), __tmp3.additional, __tmp4)
        : invalidValue(__tmp3, __tmp4);
    }
    
    if (__tmp3 === Date && typeof __tmp4 !== 'number') return transformDate(__tmp4);
    return transformPrimitive(__tmp3, __tmp4);
  }
  
  function cast<__typ3>(__tmp4: any, __tmp3: any): __typ3 {
    return transform(__tmp4, __tmp3, jsonToJSProps);
  }
  
  function uncast<__typ3>(__tmp4, __tmp3: any): any {
    return transform(__tmp4, __tmp3, jsToJSONProps);
  }
  
  function a(__tmp3: any) {
    return { arrayItems: __tmp3 };
  }
  
  function u(...typs: any[]) {
    return { unionMembers: typs };
  }
  
  //  Fran: I will be filtering out shorthands
  // function o(props: any[], additional: any) {
  //   return { props, additional };
  // }
  
  // function m(additional: any) {
  //   return { props: [], additional };
  // }

  function o(__tmp2: any[], additional: any) {
    return { props: __tmp2, additional: additional };
  }
  
  function __tmp0(additional: any) {
    return { props: [], additional:additional };
  }
  
  function r(name: __typ4) {
    return { ref: name };
  }
  
  
  const typeMap: any = {
    GrypeCvss: o(
      [
        { json: 'VendorMetadata', js: 'VendorMetadata', typ: 'any' },
        { json: 'Metrics', js: 'Metrics', typ: r('Metrics') },
        { json: 'Vector', js: 'Vector', typ: '' },
        { json: 'Version', js: 'Version', typ: '' },
      ],
      false
    ),
    Metrics: o(
      [
        { json: 'BaseScore', js: 'BaseScore', typ: 3.14 },
        { json: 'ExploitabilityScore', js: 'ExploitabilityScore', typ: 3.14 },
        { json: 'ImpactScore', js: 'ImpactScore', typ: 3.14 },
      ],
      false
    ),
  };
  