// Declare the TypeScript for Declaration Merging
// https://www.typescriptlang.org/docs/handbook/declaration-merging.html

interface Array<T> {
    diff(a: Array<T>): Array<T>;
    intersect(a: Array<T>): Array<T>;
    remove(a: Array<T> | T): Array<T>;
    add(a: Array<T> | T): Array<T>;
    equals(array: Array<T> | null): boolean;
    bubbleSort(compFunc: (
        left: T,
        right: T
    ) => number): Array<T>;
    indexOf(item: T): number;
}

// Start the javascript type patching

if (Array.prototype.diff == null) {
    Array.prototype.diff = function (a: Array<any>): Array<any> {
        return this.filter(function (i: any) {
            return !(a.indexOf(i) > -1)
        })
    }
}

if (Array.prototype.intersect == null) {
    Array.prototype.intersect = function (a: Array<any>): Array<any> {
        return this.filter(function (i) {
            return (a.indexOf(i) > -1)
        })
    }
}

if (Array.prototype.remove == null) {
    Array.prototype.remove = function (objectOrArray: Array<any> | any): Array<any> {
        if (objectOrArray == null)
            return this
        
        if (objectOrArray instanceof Array) {
            return this.diff(objectOrArray)
            
        }
        else {
            let index = this.indexOf(objectOrArray)
            if (index !== -1)
                this.splice(index, 1)
            return this
        }
    }
}

if (Array.prototype.add == null) {
    Array.prototype.add = function (objectOrArray: Array<any> | any): Array<any> {
        if (objectOrArray == null)
            return this
        
        // If for some reasons they are trying to add us to our self, throw an exception.
        if (objectOrArray === this)
            throw new Error("Array.add, I was passed myself, i can't add my self to myself.")
        
        if (objectOrArray instanceof Array) {
            for (let i = 0; i < objectOrArray.length; ++i)
                this.push(objectOrArray[i])
            return this
        }
        
        this.push(objectOrArray)
        return this
    }
}

if (Array.prototype.equals == null) {
    Array.prototype.equals = function (array: Array<any> | null): boolean {
        // if the other array is a false value, return
        if (array == null)
            return false
        
        // compare object instances
        if (this === array)
            return true
        
        // compare lengths - can save a lot of time
        if (this.length !== array.length)
            return false
        
        for (let i = 0; i < this.length; i++) {
            // Check if we have nested arrays
            if (this[i] instanceof Array && array[i] instanceof Array) {
                // recurse into the nested arrays
                if (!this[i].compare(array[i]))
                    return false
                
            }
            else if (this[i] !== array[i]) {
                // Warning - two different object instances will never be equal: {x:20} !=
                // {x:20}
                return false
                
            }
        }
        return true
    }
}

if (Array.prototype.bubbleSort == null) {
    Array.prototype.bubbleSort = function (compFunc: (
        left: any,
        right: any
    ) => number): Array<any> {
        let self = this
        
        function merge(
            left,
            right
        ) {
            let result = []
            
            while (left.length && right.length) {
                if (compFunc(left[0], right[0]) <= 0) {
                    result.push(left.shift())
                }
                else {
                    result.push(right.shift())
                }
            }
            
            while (left.length)
                result.push(left.shift())
            
            while (right.length)
                result.push(right.shift())
            
            return result
        }
        
        if (self.length < 2)
            return self.slice(0, self.length)
        
        let middle = parseInt((self.length / 2).toString())
        let left = self.slice(0, middle)
        let right = self.slice(middle, self.length)
        
        return merge(left.bubbleSort(compFunc), right.bubbleSort(compFunc))
    }
}

// ============================================================================
// Array.indexOf prptotype function
// Add if this browser (STUPID IE) doesn't support it

if (Array.prototype.indexOf == null) {
    Array.prototype.indexOf = function (item: any): <FILL> {
        let len = this.length >>> 0
        
        let from = Number(arguments[1]) || 0
        from = (from < 0) ? Math.ceil(from) : Math.floor(from)
        if (from < 0)
            from += len
        
        for (; from < len; from++) {
            if (from in this && this[from] === item)
                return from
        }
        return -1
    }
}
