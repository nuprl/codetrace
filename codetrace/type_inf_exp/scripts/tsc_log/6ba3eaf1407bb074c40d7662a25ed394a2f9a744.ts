export const getStairPermutations = (n: number, cache = new Map<number, number[][]>()): number[][] => {
    if (n === 0) {
        return []
    }

    if (cache.has(n)) {
        return cache.get(n)!
    }

    const permutations: number[][] = []
    if (n >= 3) {
        permutations.push(...getSubPermutations(n, 3, cache))
    }

    if (n >= 2) {
        permutations.push(...getSubPermutations(n, 2, cache))
    }

    if (n >= 1) {
        permutations.push(...getSubPermutations(n, 1, cache))
    }

    cache.set(n, permutations)

    return permutations
}

const getSubPermutations = (n: number, m: number, cache: Map<number, number[][]>): number[][] => {
    const perms = getStairPermutations(n - m, cache)
    if (perms.length === 0) {
        return [[m]]
    }

    return perms.map(perm => [m].concat(perm))
}




export type __typ1 = [number, number]

export const getRobotPaths = (x: number, y: number, rx: number = 0, ry: number = 0): __typ1[][] => {
    let permutations: __typ1[][] = []

    if (x === 0 || y === 0) {
        return []
    }

    
    if (x - 1 - rx > 0) {
        const paths = getRobotPaths(x, y, rx + 1, ry)
        const coord: __typ1 = [rx + 1, ry]
        const newPermutations = paths.length === 0 ? [[coord]] : paths.map(path => [coord].concat(path))
        permutations.push(...newPermutations)
    }

    
    if (y - 1 - ry > 0) {
        const paths = getRobotPaths(x, y, rx, ry + 1)
        const coord: __typ1 = [rx, ry + 1]
        const newPermutations = paths.length === 0 ? [[coord]] : paths.map(path => [coord].concat(path))
        permutations.push(...newPermutations)
    }

    if (rx === 0 && ry === 0) {
        const coord: __typ1 = [rx, ry]
        const newPermutations = permutations.length === 0 ? [[coord]] : permutations.map(path => [coord].concat(path))
        permutations = newPermutations
    }

    return permutations
}


export const findMagicIndexSlow = (numbers: number[]): number[] => {
    return numbers
        .filter((x, i) => x === i)
}

export const findMagicIndex = (numbers: number[], low: number = 0, high: number = numbers.length - 1): number | undefined => {
    if (low >= high) {
        return undefined
    }

    const midIndex = Math.floor((high + low) / 2)
    const mid = numbers[midIndex]

    if (mid === midIndex) {
        return midIndex
    }

    
    if (mid > midIndex) {
        return findMagicIndex(numbers, low, midIndex)
    }
    else {
        return findMagicIndex(numbers, midIndex + 1, high)
    }
}


export const getSubsets = (numbers: number[]): number[][] => {
    if (numbers.length <= 1) {
        return []
    }

    const first = numbers.slice(0, numbers.length - 1)
    const last = numbers.slice(1)

    return [
        first,
        last,
        ...getSubsets(first),
        ...getSubsets(last)
    ]
}


export const getPermutations = (s: __typ3): __typ3[] => {
    if (s.length <= 1) {
        return [s]
    }

    const head = s[0]
    const tail = s.slice(1)

    return getPermutations(tail)
        .map(__typ3 => [...__typ3.split('').map((x, i) => `${__typ3.slice(0, i)}${head}${__typ3.slice(i, __typ3.length)}`), `${__typ3}${head}`])
        .reduce((a, b) => a.concat(b))
}


export const parenthesesPairs = (n: number, cache = new Map<number, __typ3[]>()): __typ3[] => {
    if (n === 0) {
        return []
    }

    if (n === 1) {
        return ['()']
    }

    if (cache.has(n)) {
        return cache.get(n)!
    }

    const subPairs = parenthesesPairs(n - 1)

    const pairs = Array.from(new Set<__typ3>([
        ...subPairs.map(s => `(${s})`),
        ...subPairs.map(s => `()${s}`),
        ...subPairs.map(s => `${s}()`)
    ]).values())

    cache.set(n, pairs)

    return pairs
}



export type __typ2 = [number, number]

export const fillColor = (screen: number[][],
    point: __typ2,
    color: number
): number[][] => {
    const yMin = 0
    const yMax = screen.length

    if (yMax === 0) {
        return screen
    }

    const queue: __typ2[] = []
    const xMin = 0
    const xMax = screen[0].length
    const [px, py] = point
    const originalColor = screen[py][px]
    if (xMax === 0) {
        return screen
    }

    let firstRun = true
    const visited = new Array(screen.length).fill(() => null).map(x => new Array(screen[0].length).fill(false))

    
    queue.push(point)

    while (queue.length > 0) {
        const point = queue.shift()!
        const [x, y] = point

        if (!visited[y][x]) {
            visited[y][x] = true

            const screenColor = screen[y][x]
            
            
            if (firstRun || screenColor === originalColor) {
                screen[y][x] = color
                queue.push(...getUnvisitedNearbyPoints(screen, point, visited))
            }
        }

        firstRun = false
    }

    return screen
}

const getUnvisitedNearbyPoints = (screen: number[][], point: __typ2, visited: boolean[][]): __typ2[] => {
    const yMin = 0
    const yMax = screen.length

    if (yMax === 0) {
        return []
    }

    const xMin = 0
    const xMax = screen[0].length

    if (xMax === 0) {
        return []
    }

    const [x, y] = point

    const nearby: __typ2[] = [
        [x + 1, y],
        [x - 1, y],
        [x, y + 1],
        [x, y - 1]
    ]

    return nearby
        
        .filter(p => {
            const [px, py] = p
            return !(xMin > px
                || px >= xMax
                || yMin > py
                || py >= yMax)
                && !visited[py][px]
        })
}





export interface __typ0 {
    width: number
    height: number
    depth: number
}

export const createStackR = (boxes: __typ0[], bottom: __typ0 = null!): __typ0[] => {
    let maxHeight = 0
    let maxStack: __typ0[] = []

    boxes.forEach((box, i) => {
        if (canBeAbove(box, bottom)) {
            const otherBoxes = [...boxes]
            otherBoxes.splice(i, 1)

            const newStack = createStackR(otherBoxes, box)
            const newHeight = newStack.reduce((h, b) => { h += b.height; return h }, 0)
            if (newHeight > maxHeight) {
                maxHeight = newHeight
                maxStack = newStack
            }
        }
    })

    if (bottom !== null) {
        maxStack.unshift(bottom)
    }

    return maxStack
}

const canBeAbove = (box: __typ0, bottom: __typ0): boolean => {
    if (bottom === null) {
        return true
    }

    return box.width < bottom.width
        && box.height < bottom.height
        && box.depth < bottom.depth
}

export const createStack = (boxes: __typ0[], bottom: __typ0 = null!, cache: Map<__typ0, __typ0[]> = new Map<__typ0, __typ0[]>()): __typ0[] => {
    if (cache.has(bottom)) {
        return cache.get(bottom)!
    }

    let maxHeight = 0
    let maxStack: __typ0[] = []

    boxes.forEach((box, i) => {
        if (canBeAbove(box, bottom)) {
            const otherBoxes = [...boxes]
            otherBoxes.splice(i, 1)

            const newStack = createStack(otherBoxes, box, cache)
            const newHeight = newStack.reduce((h, b) => { h += b.height; return h }, 0)
            if (newHeight > maxHeight) {
                maxHeight = newHeight
                maxStack = newStack
            }
        }
    })

    if (bottom !== null) {
        maxStack = [bottom, ...maxStack]
        cache.set(bottom, maxStack)
    }

    return maxStack
}