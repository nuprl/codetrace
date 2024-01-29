export const Kind = 'Core#RequestTime';

/**
 * Schema definition for API Client request timings. This is mostly consistent with HAR timings.
 */
export interface IRequestTime {
  kind?: typeof Kind;
  connect: <FILL>;
  receive: number;
  send: number;
  wait: number;
  blocked: number;
  dns: number;
  ssl?: number;
}


export class RequestTime {
  kind = Kind;
  connect = -1;
  receive = -1;
  send = -1;
  wait = -1;
  blocked = -1;
  dns = -1;
  ssl?: number;

  /**
   * @param input The timings definition used to restore the state.
   */
  constructor(input?: string|IRequestTime) {
    let init: IRequestTime;
    if (typeof input === 'string') {
      init = JSON.parse(input);
    } else if (typeof input === 'object') {
      init = input;
    } else {
      init = {
        connect: -1,
        receive: -1,
        send: -1,
        wait: -1,
        blocked: -1,
        dns: -1,
      };
    }
    this.new(init);
  }

  /**
   * Creates a new timing clearing anything that is so far defined.
   */
  new(init: IRequestTime): void {
    const { connect=-1, receive=-1, send=-1, wait=-1, blocked=-1, dns=-1, ssl=-1 } = init;
    this.kind = Kind;
    this.connect = connect;
    this.receive = receive;
    this.send = send;
    this.wait = wait;
    this.blocked = blocked;
    this.dns = dns;
    this.ssl = ssl;
  }

  toJSON(): IRequestTime {
    const result: IRequestTime = {
      kind: Kind,
      connect: this.connect,
      receive: this.receive,
      send: this.send,
      wait: this.wait,
      blocked: this.blocked,
      dns: this.dns,
    };
    if (typeof this.ssl === 'number') {
      result.ssl = this.ssl;
    }
    return result;
  }

  total(): number {
    let result = 0;
    if (typeof this.connect === 'number' && this.connect > 0) {
      result += this.connect;
    }
    if (typeof this.receive === 'number' && this.receive > 0) {
      result += this.receive;
    }
    if (typeof this.send === 'number' && this.send > 0) {
      result += this.send;
    }
    if (typeof this.wait === 'number' && this.wait > 0) {
      result += this.wait;
    }
    if (typeof this.blocked === 'number' && this.blocked > 0) {
      result += this.blocked;
    }
    if (typeof this.dns === 'number' && this.dns > 0) {
      result += this.dns;
    }
    if (typeof this.ssl === 'number' && this.ssl > 0) {
      result += this.ssl;
    }
    return result;
  }
}
