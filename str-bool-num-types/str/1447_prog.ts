/* Defines the license entity */
export interface Student {
    adminNotes: <FILL>;
    appNum: string;
    appTitle: string;
    appURL: string;     
    applicationFee: number | null;
    authority: string;   
 //   authorityId: number;
    authURL: string;
    authEmail: string;
    authExt: string | null;
    authPhone: string;
    busCoordNotes: string;
    certification: string | null;   
    continuingedu: string | null;  
    coordURL: string;
    coordEmail: string;
    coordExt: string;
    coordPhone: string;
    criminal: string | null;
    duration: string;
    education: string | null;  
    exam: string | null;
    exemptions: string;
    experience: string | null;   
    id: number;
    inactive: string| null;
    keywords: string | null;
    licAuth: string;   
    licAuthId: string;   
    licDesc: string;   
    licNum: number;
    licNumType: string;   
    licTitle: string;   
    licenseId: string;   
    licenseType: string;   
    licenseUpdated: string; 
    licenseURL: string;
    markForAuthXfer: boolean;
    markForDelete: boolean;
    markForDeleteBLNC: boolean;  
    markForDeleteLEAD: boolean;
    miscFee: number;
    newLicFee: number;
    otherReq: string; 
    physical: string;  
    reasonForDelete: string;   
    reciprocity: string;   
    recipStates;
    renewalFee: number;
    ricipFee: number;
    socCode: string;   
    socDesc: string;   
    socTitle: string;
    updatedBy: string;   
    veteran: string;

    tagList: any;
    coordId: string;
    department: string;
    division: string;
    board: string;
    address1: string;
    address2: string;
    city: string;
    st: string;
    zip: string;
    multipleLocation: boolean;
    authphone: string;
    fax: string;
    contact: string;
    url: string;
    officeHours: string;
    coordName: string;
    hasOccProfile: boolean; 
  }
  
  export interface StudentResolved {
    student: any;
    error?: any;
  }

  export function newStudent() : Student {
    const l:Student = {
      adminNotes: '',     
      appNum: '',
      appTitle: '',
      appURL: '',    
      applicationFee: null,
      authority: '',  
      authEmail: '',
      authExt: '',
      authPhone: '',
      authURL: '',
      busCoordNotes: '',   
      certification: null,
      continuingedu: null,
      coordEmail: '',
      coordExt: '',
      coordPhone: '',
      coordURL: '',
      criminal: null, 
      duration: '',
      education: null, 
      exam: null,
      exemptions: '',   
      experience: null,
      id: 0,
      inactive: null,
      keywords: '',
      licAuth: '',   
      licAuthId: null,   
      licDesc: '',   
      licNum: null,
      licNumType: '',   
      licTitle: '',   
      licenseId: null, 
      licenseType: null, 
      licenseUpdated: '',
      licenseURL: '',
      markForAuthXfer: false,
      markForDelete: false,
      markForDeleteBLNC: false, 
      markForDeleteLEAD: false,
      miscFee: null,
      newLicFee: null,
      otherReq: '',
      physical: null,   
      reasonForDelete: '',  
      reciprocity: '',   
      recipStates: [],
      renewalFee: null,
      ricipFee: null,
      socCode: null,  
      socDesc: null,   
      socTitle: '',
      updatedBy: null,
      veteran: null,
      tagList: null,
      coordId: '',
      department: '',
      division: '',
      board: '',
      address1: '',
      address2: '',
      city: '',
      st: '',
      zip: '',
      multipleLocation: false,
      authphone: '',
      fax: '',
      contact: '',
      url: '',
      officeHours: '',
      coordName: '',
      hasOccProfile: false,        
    };
    return l;
  }

  export const enumFields=[
    {name: 'edu', col: 'education', label: 'Education:', sel: '--Please select'},
    {name: 'cert', col: 'certification', label: 'Certification:', sel: '--Please select'},
    {name: 'cedu', col: 'continuingedu', label: 'Continuing education:', sel: '--Please select'},
    {name: 'exp', col: 'experience', label: 'Experience:', sel: '--Please select'},
    {name: 'exam', col: 'exam', label: 'Exam:', sel: '--Please select'},
    {name: 'crim', col: 'criminal', label: 'Criminal:', sel: '--Please select'},
    {name: 'phy', col: 'physical', label: 'Physical requirements:', sel: '--Please select'},
    {name: 'vet', col: 'veteran', label: 'Veteran:', sel: '--Please select'},
    {name: 'active', col: 'inactive', label: 'Active status:', sel: '--Please select'},
    {name: 'types', col: 'licenseType', label: 'License Type:', sel: '--Please select'},
   ];  