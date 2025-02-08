import gzip
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

from docplex.mp.model import Model
from docplex.mp.format import parse_format


def export_model_to_file(model: Model, out_filename: str):
    # export a model to a path, infer format from path suffix
    #
    model._checker.typecheck_string(out_filename, caller="export_model_to_file")
    out_path = Path(out_filename)
    suffixes = out_path.suffixes
    if not suffixes:
        model.fatal("Cannot output format from path: {0}", out_filename)
    should_gzip = False
    fmt_ext = ""

    def _suffix_to_ext(ext_):
        return ext_.lstrip('.').lower()

    if len(suffixes) == 1:
        fmt_ext = _suffix_to_ext(suffixes[-1])
    else:
        if suffixes[-1] == ".gz":
            should_gzip = True
            fmt_ext = _suffix_to_ext(suffixes[-2])
    format_ = parse_format(fmt_ext)
    if not format_:
        model.fatal("Invalid format extension", fmt_ext)
    print(f"-- exporting model to format {format_.name}, gzip: {should_gzip}")
    if fmt_ext == "sav" and should_gzip:
        # force using cplex native C code for sav.gz
        model_export_name = "savgz"
        should_gzip = False
    else:
        model_export_name = fmt_ext

    export_method_name = f"export_as_{model_export_name}"
    try:
        export_method = getattr(model, export_method_name)
    except AttributeError:
        export_method = None

    # apply method
    if not export_method:
        model.fatal("Export method not found: {0}, exiting", export_method_name)

    #print(f"-- apply method {export_method_name}")
    if should_gzip:
        with NamedTemporaryFile(suffix=f".{fmt_ext}", mode="w+b", delete=False) as temp_file:
            try:
                export_method(temp_file.name)
                gzip_filename = out_filename

                with gzip.GzipFile(gzip_filename, mode="wb") as zip_file:
                    copyfileobj(temp_file, zip_file)
                print(f"-- written GZIP file {gzip_filename}")

            finally:
                temp_file.close()
                try:
                    # make sure temp file is removed, whatsoever
                    os.unlink(temp_file.name)
                except FileNotFoundError:
                    pass
            return gzip_filename
    else:
        export_method(out_filename)
        print(f"-- written output file \"{out_filename}\"")
        return out_filename


if __name__ == "__main__":
    from examples.delivery.modeling.diet import build_diet_model
    dtm = build_diet_model()
    export_model_to_file(dtm, "foo.lp.gz")