type TimeObject  = { 
    hh:number, 
    mm:number,
    am?: <FILL>
} 


declare interface Date {
    getTimeObject( options?: Intl.DateTimeFormatOptions ):TimeObject

    setTimeObject( t:TimeObject  ):void

    addMinutes( minutes:number ):Date
    
    // toLocaleTimeObjectString(locales?: string | string[]):string
    
    toTimeZoneIndependentString( options?: Intl.DateTimeFormatOptions ):string
    toTimeZoneDependentString( options?: Intl.DateTimeFormatOptions ):string
}

Date.prototype.addMinutes = function( minutes:number) {

    const mm = this.getMinutes() + minutes
    const result = new Date( this.getTime() ) 
    result.setMinutes( mm )
    return result
}

Date.prototype.getTimeObject = function( options?: Intl.DateTimeFormatOptions ):TimeObject { 

    if( options?.hour12 ) {
        const result:TimeObject = { 
            hh:this.getHours() - 12, 
            mm:this.getMinutes(), 
            am:false 
        }
    
        if( result.hh === 0 ) {
            result.hh = 12
        }
        else if( result.hh === -12 ) {
            result.hh = 12
            result.am = true
        }
        else if( result.hh < 0 ) {
            result.hh = 12 + result.hh
            result.am = true
        }
    
        return result     
    }

    return { hh:this.getHours(), mm:this.getMinutes() } 
}

Date.prototype.setTimeObject = function(t:TimeObject) { 
    this.setHours(t.hh)
    this.setMinutes(t.mm)
} 

Date.prototype.toTimeZoneIndependentString = function( options?: Intl.DateTimeFormatOptions ) {

    const twodigit = ( v:number) =>  (v < 10) ? `0${v}` : `${v}` 

    const tmpDate = this.addMinutes( this.getTimezoneOffset() )

    const t = tmpDate.getTimeObject( options )
    
    if( options?.hour12 ) {    
        return `${twodigit(t.hh)}:${twodigit(t.mm)} ${(t.am) ? 'AM': 'PM'}`
    }  

    return `${twodigit(t.hh)}:${twodigit(t.mm)}`

}

Date.prototype.toTimeZoneDependentString = function( options?: Intl.DateTimeFormatOptions ) {

    const twodigit = ( v:number) =>  (v < 10) ? `0${v}` : `${v}` 

    const t = this.getTimeObject( options )

    if( options?.hour12 ) {
        return `${twodigit(t.hh)}:${twodigit(t.mm)} ${(t.am) ? 'AM': 'PM'}`
    }
    
    return `${twodigit(t.hh)}:${twodigit(t.mm)}`
    
}


// Date.prototype.toLocaleTimeObjectString = function(locales?: string | string[]) {

//     let options: Intl.DateTimeFormatOptions = {  
//         //weekday: "short", 
//         //year: "numeric", 
//         //month: "short",  
//         //day: "numeric", 
//         hour: "2-digit", 
//         minute: "2-digit"
      
//     };  
    
//     return this.toLocaleTimeString(locales, options); 

// }