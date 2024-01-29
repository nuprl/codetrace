export const Kind = 'Core#WebApi';

/**
 * @deprecated This was used in the old version of ARC.
 */
export interface ILegacyRestApi {
  /**
   * The ID of the index item that this entry refers to.
   */
  indexId: string;
  /**
   * Version name of the API
   */
  version: string;
  /**
   * API data model. It is the output of the AMF parser run on the API.
   */
  data: string;
  /**
   * The AMF parser version used to parse this document.
   */
  amfVersion?: string;
}

export interface IWebApi {
  kind: typeof Kind;
  /**
   * The ID of the index item that this entry refers to.
   */
  indexId: string;
  /**
   * Version name of the API
   */
  version: string;
  /**
   * API data model. It is the output of the AMF parser run on the API.
   * This is left for compatibility.
   * @deprecated This was used in the old version of ARC. v18 uses the `path` with the location of the API project.
   */
  data?: string;
  /**
   * The AMF parser version used to parse this document.
   * This is left for compatibility.
   * @deprecated This was used in the old version of ARC. v18 does not use this information.
   */
  amfVersion?: string;
  /**
   * The location of the API project. This can be any URI to get the sources of the API project.
   */
  path: string;
  /**
   * Optional information to point to the API's main file.
   */
  main?: string;
  /**
   * The API format's media type.
   */
  mime?: string;
  /**
   * The API vendor. E.g. RAML 1.0, OAS 3.0, ASYNC 2.0, ...
   */
  vendor?: string;
}

export class WebApi {
  kind = Kind;
  /**
   * The ID of the index item that this entry refers to.
   */
  indexId = '';
  /**
   * Version name of the API
   */
  version = '';
  /**
   * API data model. It is the output of the AMF parser run on the API.
   * This is left for compatibility.
   * @deprecated This was used in the old version of ARC. v18 uses the `path` with the location of the API project.
   */
  data?: <FILL>;
  /**
   * The AMF parser version used to parse this document.
   * This is left for compatibility.
   * @deprecated This was used in the old version of ARC. v18 does not use this information.
   */
  amfVersion?: string;
  /**
   * The location of the API project. This can be any URI to get the sources of the API project.
   */
  path = '';
  /**
   * Optional information to point to the API's main file.
   */
  main?: string;
  /**
   * The API format's media type.
   */
  mime?: string;
  /**
   * The API vendor. E.g. RAML 1.0, OAS 3.0, ASYNC 2.0, ...
   */
  vendor?: string;

  /**
   * Checks whether the object represents a legacy way of storing the web API data.
   */
  get isLegacy(): boolean {
    return !!this.data;
  }

  /**
   * Checks whether the object is the legacy schema for web API (formally known as RestAPI)
   */
  static isLegacy(api: unknown): boolean {
    const legacy = api as ILegacyRestApi;
    if (legacy.data) {
      return true;
    }
    return false;
  }

  static fromLegacy(api: ILegacyRestApi): WebApi {
    const { version, amfVersion, data, indexId } = api;
    const init: IWebApi = {
      kind: Kind,
      version,
      path: '',
      indexId,
    };
    if (amfVersion) {
      init.amfVersion = amfVersion;
    }
    if (data) {
      init.data = data;
    }
    return new WebApi(init);
  }

  /**
   * Creates an identifier of this web API object.
   * @param indexId The id of the corresponding index item. This is usually the base URI of the API.
   * @param version The version name of this web API.
   * @returns The unique and reversible identifier of this web API.
   */
  static createId(indexId: string, version: string): string {
    return `${indexId}|${version}`;
  }

  constructor(input?: string|IWebApi) {
    let init: IWebApi;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        kind: Kind,
        version: '',
        path: '',
        indexId: '',
      };
    }
    this.new(init);
  }

  new(init: IWebApi): void {
    const { version, data, amfVersion, path, main, mime, vendor, indexId } = init;
    if (indexId) {
      this.indexId = indexId;
    } else {
      this.indexId = '';
    }
    if (version) {
      this.version = version;
    } else {
      this.version = '';
    }
    if (data) {
      this.data = data;
    } else {
      this.data = undefined;
    }
    if (amfVersion) {
      this.amfVersion = amfVersion;
    } else {
      this.amfVersion = undefined;
    }
    if (path) {
      this.path = path;
    } else {
      this.path = '';
    }
    if (main) {
      this.main = main;
    } else {
      this.main = undefined
    }
    if (mime) {
      this.mime = mime;
    } else {
      this.mime = mime;
    }
    if (vendor) {
      this.vendor = vendor;
    } else {
      this.vendor = undefined;
    }
  }

  toJSON(): IWebApi {
    const result: IWebApi = {
      kind: Kind,
      version: this.version,
      path: this.path,
      indexId: this.indexId,
    };
    if (this.data) {
      result.data = this.data;
    }
    if (this.amfVersion) {
      result.amfVersion = this.amfVersion;
    }
    if (this.main) {
      result.main = this.main;
    }
    if (this.mime) {
      result.mime = this.mime;
    }
    if (this.vendor) {
      result.vendor = this.vendor;
    }
    return result;
  }
}
