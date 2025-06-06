import pandas as pd
import altair as alt

'''
exmple:

enc_data = ['ich', 'mochte', 'ein', 'bier', 'P'] # string list
dec_data = ['I', 'want', 'a', 'bear', '.', 'E']

viz_decoder_src(model, dec_data, enc_data).save('dec_enc_attn.html')

'''

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    "convert a dense matrix to a data frame with row and column indices"
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.3d %s"
                % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.3d %s"
                % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        # if float(m[r,c]) != 0 and r < max_row and c < max_col],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(
        attn[0, head].data,
        max_dim,
        max_dim,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )

def get_encoder(model, layer):
    return model.encoder.layer_stack[layer].enc_self_attn.attention
    # return model.encoder.layers[layer].self_attn.attn_score


def get_decoder_self(model, layer):
    return model.decoder.layer_stack[layer].dec_self_attn.attention
    # return model.decoder.layers[layer].self_attn.attn_score


def get_decoder_src(model, layer):
    return model.decoder.layer_stack[layer].dec_enc_attn.attention
    # return model.decoder.layers[layer].enc_dec_attention.attn_score


def visualize_layer(model, layer, getter_fn, ntokens, row_tokens, col_tokens):
    # ntokens = last_example[0].ntokens
    attn = getter_fn(model, layer)
    n_heads = attn.shape[1]
    charts = [
        attn_map(
            attn,
            0,
            h,
            row_tokens=row_tokens,
            col_tokens=col_tokens,
            max_dim=ntokens,
        )
        for h in range(n_heads)
    ]
    assert n_heads == 8
    return alt.vconcat(
        charts[0]
        | charts[1]
        | charts[2]
        | charts[3]
        | charts[4]
        | charts[5]
        | charts[6]
        | charts[7]
        # layer + 1 due to 0-indexing
    ).properties(title="Layer %d" % (layer + 1))

def viz_encoder_self(model, enc_data):
    layer_viz = [
        visualize_layer(
            model, layer, get_encoder, len(enc_data), enc_data, enc_data
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )

def viz_decoder_self(model, dec_data):
    layer_viz = [
        visualize_layer(
            model, layer, get_decoder_self, len(dec_data), dec_data, dec_data
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )

def viz_decoder_src(model, enc_data, dec_data):
    layer_viz = [
        visualize_layer(
            model, layer, get_decoder_src, max(len(enc_data), len(dec_data)), enc_data, dec_data
        )
        for layer in range(6)
    ]
    return alt.hconcat(
        layer_viz[0]
        # & layer_viz[1]
        & layer_viz[2]
        # & layer_viz[3]
        & layer_viz[4]
        # & layer_viz[5]
    )