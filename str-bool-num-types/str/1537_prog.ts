export type Pitch = number;

export const C = 0 as Pitch;
export const CIS = 1 as Pitch;
export const D = 2 as Pitch;
export const DIS = 3 as Pitch;
export const E = 4 as Pitch;
export const F = 5 as Pitch;
export const FIS = 6 as Pitch;
export const G = 7 as Pitch;
export const GIS = 8 as Pitch;
export const A = 9 as Pitch;
export const AIS = 10 as Pitch;
export const B = 11 as Pitch;

export const NUM_PITCHES = 12;
export const PITCHES = Array.from(Array(NUM_PITCHES).keys()) as Pitch[];

export function getPitchWithOctave(index: Pitch, octave: number): string {
    const pitch = getPitch(index);
    const octaveFixed = octave + (index / NUM_PITCHES);
    return pitch + octaveFixed;
}

export function getPitch(index: number): <FILL> {
    switch (index % NUM_PITCHES) {
        case C:
            return "C";
        case CIS:
            return "C#";
        case D:
            return "D";
        case DIS:
            return "D#";
        case E:
            return "E";
        case F:
            return "F";
        case FIS:
            return "F#";
        case G:
            return "G";
        case GIS:
            return "G#";
        case A:
            return "A";
        case AIS:
            return "A#";
        case B:
            return "B";
        default:
            throw new Error("Invalid pitch");
    }
}

export function isBlack(index: number): boolean {
    return [CIS, DIS, FIS, GIS, AIS].includes(index % NUM_PITCHES);
}

