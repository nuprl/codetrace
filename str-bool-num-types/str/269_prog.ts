//
//  DocumentationPageBlockShortcut.ts
//  Pulsar Language
//
//  Created by Jiri Trecak.
//  Copyright © 2021 Supernova. All rights reserved.
//

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// MARK: - Imports

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// MARK: - Definitions

export interface DocumentationPageBlockShortcutModel {
  title?: string
  description?: string
  assetId?: string
  assetUrl?: string
  documentationItemId?: <FILL>
  url?: string
  urlPreview?: {
    title?: string
    description?: string
    thumbnailUrl?: string
  }
  documentationItemPreview?: {
    title?: string
    valid?: boolean
  }
}

export enum DocumentationPageBlockShortcutType {
  external = 'External',
  internal = 'Internal'
}

// --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
// MARK: -  Object Definition

export class DocumentationPageBlockShortcut {
  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // MARK: - Public properties

  // Visual data
  title: string | null
  description: string | null
  previewUrl: string | null

  // Linking data
  externalUrl: string | null
  internalId: string | null

  // Block type
  type: DocumentationPageBlockShortcutType

  // --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
  // MARK: - Constructor

  constructor(model: DocumentationPageBlockShortcutModel) {
    // First, configure type of the block properly
    if (model.url) {
      this.type = DocumentationPageBlockShortcutType.external
    } else {
      this.type = DocumentationPageBlockShortcutType.internal
    }

    // Configure visual block information
    this.title = this.shortcutTitleFromModel(model, this.type)
    this.description = this.shortcutDescriptionFromModel(model, this.type)
    this.previewUrl = this.shortcutPreviewUrlFromModel(model, this.type)

    // Configure linking for pages and group inside documentation
    if (
      this.type === DocumentationPageBlockShortcutType.internal &&
      model.documentationItemPreview?.valid &&
      model.documentationItemId
    ) {
      this.internalId = model.documentationItemId
    } else {
      this.internalId = null

      // Configure for external URLs
      if (this.type === DocumentationPageBlockShortcutType.external && model.url) {
        this.externalUrl = model.url
      } else {
        this.externalUrl = null
      }
    }
  }

  shortcutTitleFromModel(
    model: DocumentationPageBlockShortcutModel,
    type: DocumentationPageBlockShortcutType
  ): string | null {
    let result: string | null = null

    if (model.title && model.title.trim().length > 0) {
      result = model.title
    } else if (type === DocumentationPageBlockShortcutType.internal) {
      result = model.documentationItemPreview?.title ?? null
    } else if (type === DocumentationPageBlockShortcutType.external) {
      result = model.urlPreview?.title ?? model.url ?? null
    }

    if (!result || result.trim().length === 0) {
      return null
    }

    return result
  }

  shortcutDescriptionFromModel(
    model: DocumentationPageBlockShortcutModel,
    type: DocumentationPageBlockShortcutType
  ): string | null {
    let result: string | null = null

    if (model.description && model.description.trim().length > 0) {
      result = model.description
    } else if (type === DocumentationPageBlockShortcutType.external) {
      result = model.urlPreview?.description
    }

    if (!result || result.trim().length === 0) {
      return null
    }

    return result
  }

  shortcutPreviewUrlFromModel(
    model: DocumentationPageBlockShortcutModel,
    type: DocumentationPageBlockShortcutType
  ): string | null {
    return model.assetUrl ?? model.urlPreview?.thumbnailUrl ?? null
  }
}
