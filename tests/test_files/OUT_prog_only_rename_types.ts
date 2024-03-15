type __typ1 = number;
type __typ5 = never;
type __typ0 = string;

export interface __typ3 {
    VendorMetadata: any;
    Metrics: __typ4;
    Vector: __typ0;
    Version: __typ0;
  }
  
  export interface __typ4 {
    BaseScore: __typ1;
    ExploitabilityScore: __typ1;
    ImpactScore: __typ1;
  }
  
  const names = ['VendorMetadata', 'Metrics', 'Vector', 'Version'];
  
  export class Convert {
    public static toGrypeCvss(json: __typ0): __typ3[] {
      return cast(JSON.parse(json), a(r('GrypeCvss')));
    }
  
    public static grypeCvssToJson(value: __typ3[]): __typ0 {
      return JSON.stringify(uncast(value, a(r('GrypeCvss'))), null, 2);
    }

    public static grypeCvssToJson2(value: __typ3): __typ0 {
      return JSON.stringify(uncast(value, r('GrypeCvss')), null, 2);
    }
  }
  
  function invalidValue(typ: any, val: any, key: any = ''): __typ5 {
    if (key) {
      throw Error(`Invalid value for key "${key}". Expected type ${JSON.stringify(typ)} but got ${JSON.stringify(val)}`);
    }
    throw Error(`Invalid value ${JSON.stringify(val)} for type ${JSON.stringify(typ)}`);
  }
  
  function jsonToJSProps(typ: any): any {
    if (typ.jsonToJS === undefined) {
      const map: any = {};
      typ.props.forEach((p: any) => (map[p.json] = { key: p.js, typ: p.typ }));
      typ.jsonToJS = map;
    }
    return typ.jsonToJS;
  }
  
  function jsToJSONProps(typ: any): any {
    if (typ.jsToJSON === undefined) {
      const map: any = {};
      typ.props.forEach((p: any) => (map[p.js] = { key: p.json, typ: p.typ }));
      typ.jsToJSON = map;
    }
    return typ.jsToJSON;
  }
  
  function transform(val: any, typ: any, getProps: any, key: any = ''): any {
    function transformPrimitive(typ: __typ0, val: any): any {
      if (typeof typ === typeof val) return val;
      return invalidValue(typ, val, key);
    }
  
    function transformUnion(typs: any[], val: any): any {
      
      const l = typs.length;
      for (let i = 0; i < l; i++) {
        const typ = typs[i];
        try {
          return transform(val, typ, getProps);
        } catch (_) {}
      }
      return invalidValue(typs, val);
    }
  
    function transformEnum(cases: __typ0[], val: any): any {
      if (cases.indexOf(val) !== -1) return val;
      return invalidValue(cases, val);
    }
  
    function transformArray(typ: any, val: any): any {
      
      if (!Array.isArray(val)) return invalidValue('array', val);
      return val.map((el) => transform(el, typ, getProps));
    }
  
    function transformDate(val: any): any {
      if (val === null) {
        return null;
      }
      const d = new Date(val);
      if (isNaN(d.valueOf())) {
        return invalidValue('Date', val);
      }
      return d;
    }
  
    function transformObject(props: { [k: __typ0]: any }, additional: any, val: any): any {
      if (val === null || typeof val !== 'object' || Array.isArray(val)) {
        return invalidValue('object', val);
      }
      const result: any = {};
      Object.getOwnPropertyNames(props).forEach((key) => {
        const prop = props[key];
        const v = Object.prototype.hasOwnProperty.call(val, key) ? val[key] : undefined;
        result[prop.key] = transform(v, prop.typ, getProps, prop.key);
      });
      Object.getOwnPropertyNames(val).forEach((key) => {
        if (!Object.prototype.hasOwnProperty.call(props, key)) {
          result[key] = transform(val[key], additional, getProps, key);
        }
      });
      return result;
    }
  
    if (typ === 'any') return val;
    if (typ === null) {
      if (val === null) return val;
      return invalidValue(typ, val);
    }
    if (typ === false) return invalidValue(typ, val);
    while (typeof typ === 'object' && typ.ref !== undefined) {
      typ = typeMap[typ.ref];
    }
    if (Array.isArray(typ)) return transformEnum(typ, val);
    if (typeof typ === 'object') {
      return typ.hasOwnProperty('unionMembers')
        ? transformUnion(typ.unionMembers, val)
        : typ.hasOwnProperty('arrayItems')
        ? transformArray(typ.arrayItems, val)
        : typ.hasOwnProperty('props')
        ? transformObject(getProps(typ), typ.additional, val)
        : invalidValue(typ, val);
    }
    
    if (typ === Date && typeof val !== 'number') return transformDate(val);
    return transformPrimitive(typ, val);
  }
  
  function cast<__typ2>(val: any, typ: any): __typ2 {
    return transform(val, typ, jsonToJSProps);
  }
  
  function uncast<__typ2>(val: __typ2, typ: any): any {
    return transform(val, typ, jsToJSONProps);
  }
  
  function a(typ: any) {
    return { arrayItems: typ };
  }
  
  function u(...typs: any[]) {
    return { unionMembers: typs };
  }
  
  function o(props: any[], additional: any) {
    return { props, additional }; // F: I will be filtering out shorthands
  }
  
  function m(additional: any) {
    return { props: [], additional };
  }
  
  function r(name: __typ0) {
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
  