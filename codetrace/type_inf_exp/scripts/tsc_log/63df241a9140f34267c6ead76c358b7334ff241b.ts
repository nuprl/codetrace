interface Array<T> {
    get(index: number): T | undefined;
    set(index: number, element: T): T[];
    remove(index: number): T[];
    delete(index: number): T[];
    isEmpty(): boolean;
    isNotEmpty(): boolean;
    clear(): void;
    clone(): T[];
    sum(): number;
    min(): number;
    max(): number;
    first(): T | undefined;
    last(): T | undefined;
}

if (!Array.prototype.get) {
    Array.prototype.get = function <T>(
        this: T[],
        index: number
    ): T | undefined {
        return this[index];
    };
}

if (!Array.prototype.set) {
    Array.prototype.set = function <T>(
        this: T[],
        index: number,
        element: T
    ): T[] {
        this[index] = element;

        return this;
    };
}

if (!Array.prototype.remove) {
    Array.prototype.remove = function <T>(this: T[], index: number): T[] {
        return this.splice(index, 1);
    };
}

if (!Array.prototype.delete) {
    Array.prototype.delete = function <T>(this: T[], index: number): T[] {
        this.splice(index, 1);

        return this;
    };
}

if (!Array.prototype.isEmpty) {
    Array.prototype.isEmpty = function <T>(this: T[]): boolean {
        return this.length === 0;
    };
}

if (!Array.prototype.isNotEmpty) {
    Array.prototype.isNotEmpty = function <T>(this: T[]): boolean {
        return !this.isEmpty();
    };
}

if (!Array.prototype.clear) {
    Array.prototype.clear = function <T>(this: T[]): void {
        this.length = 0;
    };
}

if (!Array.prototype.clone) {
    Array.prototype.clone = function <T>(this: T[]): T[] {
        return this.slice(0);
    };
}

if (!Array.prototype.sum) {
    Array.prototype.sum = function (this: number[]): number {
        return this.reduce(
            (partialSum: number, element: number) => partialSum + element
        );
    };
}

if (!Array.prototype.min) {
    Array.prototype.min = function (this: number[]): number {
        return Math.min.apply(null, this);
    };
}

if (!Array.prototype.max) {
    Array.prototype.max = function (this: number[]): number {
        return Math.max.apply(null, this);
    };
}

if (!Array.prototype.first) {
    Array.prototype.first = function <T>(this: T[]): T | undefined {
        return this.get(0);
    };
}

if (!Array.prototype.last) {
    Array.prototype.last = function <T>(this: T[]): T | undefined {
        const length: number = this.length;

        if (length > 0) {
            return this.get(this.length - 1);
        } else {
            return undefined;
        }
    };
}