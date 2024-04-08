class __typ0 extends Symbol {};
type __typ1 = string;

export interface IEnventBus {
    publish(eventType: __typ1, agrs: number): void;
    subcribe(eventType: __typ1, callback: any): any;
}

export class Subcriber {
    private id: __typ0;
    private eventType: __typ1;
    public callback: any;
    private unSubcribe: any;

    constructor(id: __typ0, eventType: __typ1, callback: any, unSubcribe: any) {
        this.id = id;
        this.eventType = eventType;
        this.callback = callback;
        this.unSubcribe = unSubcribe;
    }

    unsubcribe() {
        this.unSubcribe(this.id, this.eventType);
    }
}

export default class EventBus implements IEnventBus {

    private subscriptions: Map<__typ1, Subcriber[]>;

    public constructor() {
        this.subscriptions = new Map();
    }

    public publish(eventType: __typ1, agrs: any) {
        if (!this.subscriptions.has(eventType)) return;

        this.subscriptions.get(eventType)!.forEach(sub => {
            sub.callback(agrs);
        });
    }

    public subcribe(eventType: __typ1, callback: any): any {
        const id = __typ0(eventType);

        const subcriber = new Subcriber(id, eventType, callback, (idUnSubcriber: any, eventTypeSubcriber: any) => {
            for (let i = 0; i < this.subscriptions.get(eventTypeSubcriber)!.length; i++) {
                if (idUnSubcriber === id) {
                    this.subscriptions.get(eventTypeSubcriber)!.splice(i, 1);
                }
            }
        });

        if (this.subscriptions.has(eventType)) {
            this.subscriptions.get(eventType)!.push(subcriber);
        } else {
            this.subscriptions.set(eventType, [subcriber]);
        }
        return subcriber;
    }

    public getSubcribers() {
        return this.subscriptions;
    }
}