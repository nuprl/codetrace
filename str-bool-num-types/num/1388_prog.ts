/**
 *
 *With cursor pos
 *
 **/


type Params = {
    param: string,
    functionName: string,
    pos: number,
    functionEndPos?: number,
    functionStartPos: number,
    parameterNumber: number
}

type Positions = {
    start: number, end: number
}

type Stack = {
    functionStartPos: number,
    functionEndPos?: number,
    value: string,
    functionName: string
}

function isAllowedNameLetter(str: string) {
    return str && str.length === 1 && (str.match(/[a-z]/i) || str === "_");
}

function getName(value: string) {
    let name = "";
    let i = value.length;
    for (let i = value.length - 1; i >= 0; i--) {
        if (!isAllowedNameLetter(value[i])) break;
        name += value[i]
    }
    return name.split("").reverse().join("");
}

function isPartOf(pos: number, positions: Positions[]) {
    for (let i = 0; i < positions.length; i++) {
        if (positions[i].start <= pos && positions[i].end >= pos)
            return true;
    }
    return false;
}

function flatDeep(arr: any[], d = Infinity): any[] {
    return d > 0 ? arr.reduce((acc, val) =>
        acc.concat(Array.isArray(val) ? flatDeep(val, d - 1) : val), []) : arr.slice();
};

function getStringStartAndEndPos(value: string) {
    const doubleQuoteStringStack: { value: string, pos: number }[] = []
    const singleQuoteStringStack: { value: string, pos: number }[] = []
    const strings: { start: number, end: number }[] = [];
    for (let i = 0; i < value.length;) {
        if ((value[i] === "\\" && value[i + 1] === "\"") ||
            (value[i] === "\\" && value[i + 1] === "\'")) {
            i += 2;
            continue;
        }
        if (value[i] === "\"") {
            if (!singleQuoteStringStack.length)
                if (doubleQuoteStringStack.length) {
                    const x = doubleQuoteStringStack.pop();
                    strings.push({ start: x.pos, end: i })
                } else {
                    doubleQuoteStringStack.push({ value: value[i], pos: i })
                }
        }
        else if (value[i] === "\'") {
            if (!doubleQuoteStringStack.length)
                if (singleQuoteStringStack.length) {
                    const x = singleQuoteStringStack.pop();
                    strings.push({ start: x.pos, end: i })
                } else {
                    singleQuoteStringStack.push({ value: value[i], pos: i })
                }
        }
        i++;
    }
    return strings;
}

function getChipStartAndEndPos(value: string) {
    const chipStack: { value: string, pos: number }[] = []
    const chips: { start: number, end: number }[] = [];
    for (let i = 0; i < value.length;) {
        if (value[i] === "\ufff9") {
            if (chipStack.length) {
                const x = chipStack.pop();
                chips.push({ start: x.pos, end: i })
            } else {
                chipStack.push({ value: value[i], pos: i })
            }
        }
        i++;
    }
    return chips;
}

function getObjectStartEndPositions(value: string, stringPositions: Positions[],
    chipPositions: Positions[]) {
    const objectStack: { value: string, pos: number }[] = []
    const objects: { start: number, end: number }[] = [];
    for (let i = 0; i < value.length;) {
        if (!isPartOf(i, stringPositions) &&
            !isPartOf(i, chipPositions)) {
            if (value[i] === '{') {
                objectStack.push({ value: value[i], pos: i })
            }
            if (value[i] === '}') {
                if (objectStack.length) {
                    const x = objectStack.pop();
                    objects.push({ start: x.pos, end: i })
                }
            }
        }
        i++;
    }
    return objects;
}

function getArrayStartEndPositions(value: string, stringPositions: Positions[],
    chipPositions: Positions[]) {
    const arrayStack: { value: string, pos: number }[] = []
    const arrays: { start: number, end: number }[] = [];
    for (let i = 0; i < value.length;) {
        if (!isPartOf(i, stringPositions) &&
            !isPartOf(i, chipPositions)) {
            if (value[i] === '[') {
                arrayStack.push({ value: value[i], pos: i })
            }
            if (value[i] === ']') {
                if (arrayStack.length) {
                    const x = arrayStack.pop();
                    arrays.push({ start: x.pos, end: i })
                }
            }
        }
        i++;
    }
    return arrays;
}

function getParametersWithParamterNumber(startPos: number, value: string,
    stack: Stack[] = [], funcName: string = "", stringPositions: Positions[], chipPositions: Positions[],
    objectPositions: Positions[], arrayPositions: Positions[])
    : { params: Params[], lastVisitedCharacter: number } {
    let params: Params[] = [];
    let parameter = "";
    let pos = startPos;
    let i = startPos;
    let parameterNumber = 0;
    for (i = startPos; i < value.length; i++) {
        if (!isPartOf(i, stringPositions) && !isPartOf(i, chipPositions)
            && !isPartOf(i, objectPositions) && !isPartOf(i, arrayPositions)) {
            if (value[i] === '(') {
                let funcName = getName(value.substr(startPos, i - startPos))
                stack.push({ value: value[i], functionStartPos: i - (funcName.length), functionName: funcName })
                const sub = getParametersWithParamterNumber(i + 1, value, stack, funcName, stringPositions, chipPositions,
                    objectPositions, arrayPositions)
                parameterNumber++
                params = [...params, ...sub.params]
                i = sub.lastVisitedCharacter + 1;
                parameter = "";
                pos = i;
                continue;
            }
            else if (value[i] === ')') {
                const endPos = stack.pop();
                if (endPos)
                    params.push({
                        param: parameter,
                        functionName: funcName, pos,
                        functionEndPos: i,
                        functionStartPos: endPos.functionStartPos,
                        parameterNumber
                    })
                return { params: params, lastVisitedCharacter: i };
            }
            else if (value[i] === ',') {
                if (parameter) {
                    params.push({
                        param: parameter, functionName: funcName, pos: pos,
                        functionStartPos: stack[stack.length - 1]?.functionStartPos,
                        parameterNumber
                    });
                    parameter = "";
                    pos = i + 1;
                }
                parameterNumber++;
                continue;
            }
        }
        parameter += value[i];
    }
    return { params: params, lastVisitedCharacter: i };
}

function getParameterToFunctionMappingList(value: string, stringPositions: Positions[],
    chipPositions: Positions[], objectPositions: Positions[], arrayPositions: Positions[]): Params[] {
    let params: Params[] = [];
    let startPos = 0;
    let stack: Stack[] = [];
    while (startPos < value.length) {
        let newParams = getParametersWithParamterNumber(startPos, value, stack, "", stringPositions,
            chipPositions, objectPositions, arrayPositions)
        params = [...params, ...newParams.params];
        startPos = newParams.lastVisitedCharacter + 1;
    }
    stack.forEach(ele => {
        params.push({
            functionName: ele.functionName,
            functionStartPos: ele.functionStartPos,
            param: "",
            parameterNumber: 0,
            pos: ele.functionStartPos,
        })
    })
    return flatDeep(params)
        .sort((a: Params, b: Params) => (a.functionStartPos > b.functionStartPos ? 1
            : (a.functionStartPos < b.functionStartPos) ? -1 : 0));
}



export function getFunctionNameFromCursorPos(value: string, cursor: number) {
    /* 
    * StringPositions contain all the strings which are part of the value along with their start and end pos
    */
    const stringPositions = getStringStartAndEndPos(value)
    /*
    * Chip positions
    */
    const chipPositions = getChipStartAndEndPos(value);
    /**
     * Object positions
     */
    const objectPositions = getObjectStartEndPositions(value, stringPositions, chipPositions);
    /**
     * Array positions
     */
    const arrayPositions = getArrayStartEndPositions(value, stringPositions, chipPositions)

    const paramsList = getParameterToFunctionMappingList(value, stringPositions, chipPositions, objectPositions,
        arrayPositions);
    let funcStartPos = -Infinity;
    let funcEndPos = -Infinity;
    let ranges: { [start: number]: <FILL> } = {}
    paramsList.forEach(param => {
        ranges[param.functionStartPos] = param.functionEndPos
    })
    const updatedParamList = paramsList.map(param => {
        return {
            ...param,
            functionEndPos: ranges[param.functionStartPos]
        }
    })
    updatedParamList.forEach(param => {
        if (param.functionStartPos <= cursor && param.functionEndPos && param.functionEndPos >= cursor) {
            funcStartPos = param.functionStartPos;
            funcEndPos = param.functionEndPos
        }
    });
    updatedParamList.filter(param => param.functionEndPos === undefined).forEach(param => {
        if (param.functionStartPos >= funcStartPos && param.functionStartPos <= cursor) {
            funcStartPos = param.functionStartPos;
        }
    })
    const allParameters = updatedParamList.filter(param =>
        param.functionStartPos === funcStartPos);
    const functionName = allParameters.length ? allParameters[0].functionName : "";
    //parameterNumber is the parameterNumber of the nearest parameter to the left
    let parameterNumber = -Infinity;
    let pos = -Infinity;
    allParameters.forEach(param => {
        if (param.pos <= cursor && param.pos > pos) {
            parameterNumber = param.parameterNumber
            pos = param.pos;
        }
    })
    if (pos !== -Infinity) {
        for (let i = pos + 1; i < cursor; i++) {
            if (value[i] === ',' &&
                !isPartOf(i, stringPositions) &&
                !isPartOf(i, chipPositions) &&
                !isPartOf(i, objectPositions) &&
                !isPartOf(i, arrayPositions)) {
                parameterNumber++
            }
        }
    }
    return { functionName, parameterNumber }
}

// console.log((getParametersWithCursorPos(0, ' sum(a,sub(b, x),c,div(const(d),y),x) + 2 + mul(1,2)').params[0]).flat(Infinity).sort((a, b) => (a.startPos > b.startPos ? 1 : (a.startPos < b.startPos) ? -1 : 0)))

// console.log((getFunctions(' sum(a,sub(b,x),c,div(const(d),y),x) + 2 + mul(1,2)')))

// console.log((getParameterToFunctionMappingList(' sum(a,sub(b,x),c,div(const(d),y),x) + 2 + mul(1,  now()')))
// console.log((getFunctionNameFromCursorPos(' sum(a,sub(b,x),c,div(const(d),y),x) + 2 + mul(1,2)  now()', 49)))
