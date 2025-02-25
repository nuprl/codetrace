export type TLinkedListItem<T> = {
    data: T;
    next: TLinkedListItem<T>;
};

export type TLinkedListOptions<T> = {
    
    equals?: (left: T, right: T) => Promise<boolean>;
};


export class LinkedList<T> {

    
    protected itemsCount: number = 0;

    
    protected equals: (left: T, right: T) => Promise<boolean>;

    
    protected rootItem: TLinkedListItem<T>;

    constructor( options?: TLinkedListOptions<T> ) {
        this.equals = (options && options.equals) || (async(left: <FILL>, right: T ) => {
            return left == right;
         });
    }

    
    public async replace(data: T, replacer: T, appendIfNotFound:boolean = false): Promise<boolean> {

        let tail = await this.browseList( async (item: TLinkedListItem<T>)=> {

            let areEquals = await this.equals(item.data, data);

            if(areEquals){
                item.data = replacer;
                return true;
            }

        });

        if(tail && appendIfNotFound){
            
            tail.next = {
                data: replacer,
                next: null
            };

            this.itemsCount++;

            return true;
        }
        
        
        return !tail;
    }

    
    public async insert(data: T, before?: T): Promise<void> {

        let newItem: TLinkedListItem<T> = {
            data: data,
            next: null
        };

        if(!this.rootItem){
            this.rootItem = newItem;
            this.itemsCount++;
            return;
        }

        
        let areEquals: boolean;

        
        if(before != undefined){
            areEquals = await this.equals(this.rootItem.data, before);
            if(areEquals){
                let next = this.rootItem;
                this.rootItem = newItem;
                newItem.next = next;

                this.itemsCount++;
                return;
            }
        }
        
        let tail = await this.browseList( async (item: TLinkedListItem<T>) => {

            if(before != undefined && item.next) {
                areEquals = await this.equals(item.next.data, before);
                if(areEquals){
                    let next = item.next;
                    item.next = newItem;
                    newItem.next = next;
                    return true;
                }
            }

        });

        if(tail){
            
            tail.next = newItem;
        }
        
        this.itemsCount++;
    }

    
    public getIterator(): { next: () => T; hasNext: () => boolean; } {

        
        let current: TLinkedListItem<T> = this.rootItem ? this.rootItem : null,
            hasNext: boolean = this.rootItem ? true : false;

        return {

            
            next: () => {
                
                if(!hasNext){
                    return null;
                }

                let result: T = current.data;

                current = current.next;

                hasNext = current != null;

                return result;
            },

            
            hasNext: () => {
                return hasNext;
            }
        };
    }

    
    public async delete(data: T): Promise<T> {
        if(!this.rootItem) {
            return undefined;
        }

        let result: T;

        
        let areEquals: boolean = await this.equals(this.rootItem.data, data);

        if(areEquals){
            result = this.rootItem.data;
            this.rootItem = this.rootItem.next;
            this.itemsCount--;
            return result;
        }

        let tail = await this.browseList( async(item: TLinkedListItem<T>) => {
            if(item.next){
                areEquals = await this.equals(item.next.data, data);
                if(areEquals){
                    
                    result = item.next.data;

                    item.next = item.next.next;
                    this.itemsCount--
                    return true;
                }
            }
        });
        
        return result;
    }

    
    public getHead(): T {
        return this.rootItem ? this.rootItem.data : null;
    }

    
    private async browseList(callback: (item: TLinkedListItem<T>) => Promise<boolean | void>): Promise<TLinkedListItem<T>> {

        if(!this.rootItem){
            return;
        }

        let current = this.rootItem;

        while(current){
            let result = await callback(current);

            if(result){
                
                
                return null;
            }

            current = current.next;
        }

        
        
        return current;

    }

    
    public getCount(): number {
        return this.itemsCount;
    }

}
